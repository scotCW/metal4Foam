// metal4Foam: FP32 compute kernels for OpenFOAM lduMatrix Krylov solvers.
//
// The sparse matrix stays in OpenFOAM's native LDU (face-addressed) layout:
//   row i of A*x = diag[i]*x[i]
//                + sum over faces f owned by i        : upper[f]*x[upp[f]]
//                + sum over faces f with neighbour i  : lower[f]*x[low[f]]
// Owner faces of cell i are contiguous  [ownStart[i], ownStart[i+1]).
// Neighbour faces are indirected through losort[losortStart[i]..].
//
// All reductions are two-stage: simd_sum within simdgroups, then a
// second simd_sum over per-simdgroup partials; one float per threadgroup
// is written to `partials`, which a single-threadgroup `reduce_scalars`
// pass folds into the scalars buffer (alpha/beta/omega/residual).

#include <metal_stdlib>
using namespace metal;

#define TG 256u

// Scalars buffer layout
#define S_WARA     0
#define S_WARAOLD  1
#define S_WAPA     2
#define S_ALPHA    3
#define S_BETA     4
#define S_RES      5
#define S_RA0RA    6
#define S_RA0RAOLD 7
#define S_RA0AYA   8
#define S_OMEGA    9
#define S_TATA     10
#define S_TASA     11

static inline float tg_sum
(
    float v,
    threadgroup float* sh,
    ushort lane,
    ushort sgid,
    ushort nsg
)
{
    v = simd_sum(v);
    if (lane == 0) sh[sgid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0)
    {
        v = (lane < nsg) ? sh[lane] : 0.0f;
        v = simd_sum(v);
    }
    return v;  // valid for sgid==0
}

#define REDUCE_ARGS \
    ushort lane [[thread_index_in_simdgroup]], \
    ushort sgid [[simdgroup_index_in_threadgroup]], \
    ushort nsg  [[simdgroups_per_threadgroup]], \
    ushort lid  [[thread_index_in_threadgroup]], \
    uint   tgid [[threadgroup_position_in_grid]]


// y = A*x in LDU form, fused with partial reduction of dot(w, y).
// For PCG pass w == x (gives pA.AyA); for BiCGStab pass w = rA0 or sA.
kernel void ldu_spmv_dotw
(
    device const float* diag        [[buffer(0)]],
    device const float* upperCoeff  [[buffer(1)]],
    device const float* lowerCoeff  [[buffer(2)]],
    device const uint*  lowAddr     [[buffer(3)]],
    device const uint*  uppAddr     [[buffer(4)]],
    device const uint*  ownStart    [[buffer(5)]],
    device const uint*  losort      [[buffer(6)]],
    device const uint*  losortStart [[buffer(7)]],
    device const float* x           [[buffer(8)]],
    device const float* w           [[buffer(9)]],
    device float*       y           [[buffer(10)]],
    device float*       partials    [[buffer(11)]],
    constant uint&      n           [[buffer(12)]],
    uint gid [[thread_position_in_grid]],
    REDUCE_ARGS
)
{
    threadgroup float sh[32];
    float dotv = 0.0f;
    if (gid < n)
    {
        float sum = diag[gid]*x[gid];

        const uint fs = ownStart[gid];
        const uint fe = ownStart[gid + 1];
        for (uint f = fs; f < fe; ++f)
        {
            sum += upperCoeff[f]*x[uppAddr[f]];
        }

        const uint ls = losortStart[gid];
        const uint le = losortStart[gid + 1];
        for (uint k = ls; k < le; ++k)
        {
            const uint f = losort[k];
            sum += lowerCoeff[f]*x[lowAddr[f]];
        }

        y[gid] = sum;
        dotv = w[gid]*sum;
    }
    const float tot = tg_sum(dotv, sh, lane, sgid, nsg);
    if (lid == 0) partials[tgid] = tot;
}


// y = A*x in LDU form, no fused dot (Chebyshev inner multiply)
kernel void ldu_spmv
(
    device const float* diag        [[buffer(0)]],
    device const float* upperCoeff  [[buffer(1)]],
    device const float* lowerCoeff  [[buffer(2)]],
    device const uint*  lowAddr     [[buffer(3)]],
    device const uint*  uppAddr     [[buffer(4)]],
    device const uint*  ownStart    [[buffer(5)]],
    device const uint*  losort      [[buffer(6)]],
    device const uint*  losortStart [[buffer(7)]],
    device const float* x           [[buffer(8)]],
    device float*       y           [[buffer(9)]],
    constant uint&      n           [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid < n)
    {
        float sum = diag[gid]*x[gid];

        const uint fs = ownStart[gid];
        const uint fe = ownStart[gid + 1];
        for (uint f = fs; f < fe; ++f)
        {
            sum += upperCoeff[f]*x[uppAddr[f]];
        }

        const uint ls = losortStart[gid];
        const uint le = losortStart[gid + 1];
        for (uint k = ls; k < le; ++k)
        {
            const uint f = losort[k];
            sum += lowerCoeff[f]*x[lowAddr[f]];
        }

        y[gid] = sum;
    }
}


// Chebyshev preconditioner, first step: w = c1*rD*r ; z = w
kernel void cheb_init
(
    device const float* r  [[buffer(0)]],
    device const float* rD [[buffer(1)]],
    device float*       w  [[buffer(2)]],
    device float*       z  [[buffer(3)]],
    constant float&     c1 [[buffer(4)]],
    constant uint&      n  [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid < n)
    {
        const float wi = c1*rD[gid]*r[gid];
        w[gid] = wi;
        z[gid] = wi;
    }
}


// Chebyshev step: w = c2*w + c3*rD*(r - v) ; z += w   (v = A*z_prev)
kernel void cheb_step
(
    device const float* r  [[buffer(0)]],
    device const float* v  [[buffer(1)]],
    device const float* rD [[buffer(2)]],
    device float*       w  [[buffer(3)]],
    device float*       z  [[buffer(4)]],
    constant float&     c2 [[buffer(5)]],
    constant float&     c3 [[buffer(6)]],
    constant uint&      n  [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid < n)
    {
        const float wi = c2*w[gid] + c3*rD[gid]*(r[gid] - v[gid]);
        w[gid] = wi;
        z[gid] += wi;
    }
}


// out = a*b elementwise (Jacobi preconditioner apply)
kernel void emul
(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid < n) out[gid] = a[gid]*b[gid];
}


// PCG: wA = rD*rA fused with partials of dot(wA, rA)
kernel void pcg_precond_dot
(
    device const float* rA       [[buffer(0)]],
    device const float* rD       [[buffer(1)]],
    device float*       wA       [[buffer(2)]],
    device float*       partials [[buffer(3)]],
    constant uint&      n        [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    REDUCE_ARGS
)
{
    threadgroup float sh[32];
    float dotv = 0.0f;
    if (gid < n)
    {
        const float wv = rA[gid]*rD[gid];
        wA[gid] = wv;
        dotv = wv*rA[gid];
    }
    const float tot = tg_sum(dotv, sh, lane, sgid, nsg);
    if (lid == 0) partials[tgid] = tot;
}


// PCG: p = wA + beta*p  (beta == 0 on first iteration; p pre-zeroed)
kernel void pcg_update_p
(
    device float*       p  [[buffer(0)]],
    device const float* wA [[buffer(1)]],
    device const float* s  [[buffer(2)]],
    constant uint&      n  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid < n) p[gid] = wA[gid] + s[S_BETA]*p[gid];
}


// PCG: delta += alpha*p; rA -= alpha*wA; partials of sum|rA|
kernel void pcg_update_sol_res
(
    device float*       delta    [[buffer(0)]],
    device float*       rA       [[buffer(1)]],
    device const float* p        [[buffer(2)]],
    device const float* wA       [[buffer(3)]],
    device const float* s        [[buffer(4)]],
    device float*       partials [[buffer(5)]],
    constant uint&      n        [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    REDUCE_ARGS
)
{
    threadgroup float sh[32];
    float m = 0.0f;
    if (gid < n)
    {
        const float a = s[S_ALPHA];
        delta[gid] += a*p[gid];
        const float r = rA[gid] - a*wA[gid];
        rA[gid] = r;
        m = fabs(r);
    }
    const float tot = tg_sum(m, sh, lane, sgid, nsg);
    if (lid == 0) partials[tgid] = tot;
}


// Generic partial dot(a, b)
kernel void dot_partial
(
    device const float* a        [[buffer(0)]],
    device const float* b        [[buffer(1)]],
    device float*       partials [[buffer(2)]],
    constant uint&      n        [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    REDUCE_ARGS
)
{
    threadgroup float sh[32];
    float v = 0.0f;
    if (gid < n) v = a[gid]*b[gid];
    const float tot = tg_sum(v, sh, lane, sgid, nsg);
    if (lid == 0) partials[tgid] = tot;
}


// Single-threadgroup pass folding threadgroup partials into solver scalars.
// mode 0: PCG   wArA + beta        (P1)
// mode 1: PCG   wApA + alpha       (P1)
// mode 2:       residual sum       (P1)
// mode 3: BiCG  rA0rA + beta       (P1)
// mode 4: BiCG  rA0AyA + alpha     (P1)
// mode 5: BiCG  omega = tAsA/tAtA  (P1 = tAtA partials, P2 = tAsA partials)
kernel void reduce_scalars
(
    device const float* P1        [[buffer(0)]],
    device const float* P2        [[buffer(1)]],
    device float*       s         [[buffer(2)]],
    constant uint&      count     [[buffer(3)]],
    constant uint&      mode      [[buffer(4)]],
    constant uint&      firstIter [[buffer(5)]],
    REDUCE_ARGS
)
{
    threadgroup float sh[32];
    float v1 = 0.0f;
    float v2 = 0.0f;
    for (uint i = lid; i < count; i += TG)
    {
        v1 += P1[i];
        if (mode == 5) v2 += P2[i];
    }
    const float t1 = tg_sum(v1, sh, lane, sgid, nsg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t2 = 0.0f;
    if (mode == 5)
    {
        t2 = tg_sum(v2, sh, lane, sgid, nsg);
    }

    if (lid == 0)
    {
        switch (mode)
        {
            case 0:
                s[S_WARAOLD] = s[S_WARA];
                s[S_WARA] = t1;
                s[S_BETA] = (firstIter != 0) ? 0.0f : t1/s[S_WARAOLD];
                break;
            case 1:
                s[S_WAPA] = t1;
                s[S_ALPHA] = s[S_WARA]/t1;
                break;
            case 2:
                s[S_RES] = t1;
                break;
            case 3:
                s[S_RA0RAOLD] = s[S_RA0RA];
                s[S_RA0RA] = t1;
                s[S_BETA] =
                    (firstIter != 0)
                  ? 0.0f
                  : (t1/s[S_RA0RAOLD])*(s[S_ALPHA]/s[S_OMEGA]);
                break;
            case 4:
                s[S_RA0AYA] = t1;
                s[S_ALPHA] = s[S_RA0RA]/t1;
                break;
            case 5:
                s[S_TATA] = t1;
                s[S_TASA] = t2;
                s[S_OMEGA] = t2/t1;
                break;
        }
    }
}


// BiCGStab: p = rA + beta*(p - omega*AyA)
// (first iteration: beta==0, and p/AyA are pre-zeroed, so p = rA)
kernel void bicg_update_p
(
    device float*       p   [[buffer(0)]],
    device const float* rA  [[buffer(1)]],
    device const float* AyA [[buffer(2)]],
    device const float* s   [[buffer(3)]],
    constant uint&      n   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid < n)
    {
        p[gid] = rA[gid] + s[S_BETA]*(p[gid] - s[S_OMEGA]*AyA[gid]);
    }
}


// BiCGStab: sA = rA - alpha*AyA; partials of sum|sA|
kernel void bicg_s_res
(
    device float*       sA       [[buffer(0)]],
    device const float* rA       [[buffer(1)]],
    device const float* AyA      [[buffer(2)]],
    device const float* s        [[buffer(3)]],
    device float*       partials [[buffer(4)]],
    constant uint&      n        [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    REDUCE_ARGS
)
{
    threadgroup float sh[32];
    float m = 0.0f;
    if (gid < n)
    {
        const float sv = rA[gid] - s[S_ALPHA]*AyA[gid];
        sA[gid] = sv;
        m = fabs(sv);
    }
    const float tot = tg_sum(m, sh, lane, sgid, nsg);
    if (lid == 0) partials[tgid] = tot;
}


// BiCGStab early exit: delta += alpha*yA
kernel void bicg_early_update
(
    device float*       delta [[buffer(0)]],
    device const float* yA    [[buffer(1)]],
    device const float* s     [[buffer(2)]],
    constant uint&      n     [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid < n) delta[gid] += s[S_ALPHA]*yA[gid];
}


// BiCGStab: delta += alpha*yA + omega*zA; rA = sA - omega*tA; partials |rA|
kernel void bicg_update_sol_res
(
    device float*       delta    [[buffer(0)]],
    device const float* yA       [[buffer(1)]],
    device const float* zA       [[buffer(2)]],
    device float*       rA       [[buffer(3)]],
    device const float* sA       [[buffer(4)]],
    device const float* tA       [[buffer(5)]],
    device const float* s        [[buffer(6)]],
    device float*       partials [[buffer(7)]],
    constant uint&      n        [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    REDUCE_ARGS
)
{
    threadgroup float sh[32];
    float m = 0.0f;
    if (gid < n)
    {
        const float a = s[S_ALPHA];
        const float o = s[S_OMEGA];
        delta[gid] += a*yA[gid] + o*zA[gid];
        const float r = sA[gid] - o*tA[gid];
        rA[gid] = r;
        m = fabs(r);
    }
    const float tot = tg_sum(m, sh, lane, sgid, nsg);
    if (lid == 0) partials[tgid] = tot;
}
