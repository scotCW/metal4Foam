/*---------------------------------------------------------------------------*\
    metal4Foam: Metal-accelerated linear solvers for OpenFOAM (Apple GPU)

    Copyright (C) 2026 metal4Foam authors
    SPDX-License-Identifier: GPL-3.0-or-later

    Implementation of the Metal backend. This is the only translation unit
    that includes metal-cpp; it is plain C++17 (no Objective-C++ needed).

\*---------------------------------------------------------------------------*/

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include <Accelerate/Accelerate.h>

#include "metalFoamBackend.H"
#include "metalFoamKernelsMSL.H"   // generated: embedded MSL source

#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fenv.h>

// Scalars buffer layout -- keep in sync with metalFoamKernels.metal
enum
{
    S_WARA = 0, S_WARAOLD, S_WAPA, S_ALPHA, S_BETA, S_RES,
    S_RA0RA, S_RA0RAOLD, S_RA0AYA, S_OMEGA, S_TATA, S_TASA,
    S_COUNT = 16
};

static constexpr uint32_t TG = 256;

namespace
{

double nowSeconds()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// OpenFOAM traps SIGFPE (FOAM_SIGFPE); the Metal/AGX driver performs FP
// operations that legitimately raise exceptions during device init and
// command submission. Restore the default (non-trapping) FP environment
// around Metal calls; trap state is per-thread and is restored on exit.
struct FpTrapGuard
{
    fenv_t saved;
    FpTrapGuard() { fegetenv(&saved); fesetenv(FE_DFL_ENV); }
    ~FpTrapGuard() { fesetenv(&saved); }
};

} // anonymous namespace


// * * * * * * * * * * * * * * * * * Impl * * * * * * * * * * * * * * * * * //

struct Foam::metalFoamBackend::Impl
{
    MTL::Device* device = nullptr;
    MTL::CommandQueue* queue = nullptr;
    MTL::Library* library = nullptr;

    MTL::ComputePipelineState* pSpmvDotw = nullptr;
    MTL::ComputePipelineState* pSpmv = nullptr;
    MTL::ComputePipelineState* pChebInit = nullptr;
    MTL::ComputePipelineState* pChebStep = nullptr;
    MTL::ComputePipelineState* pEmul = nullptr;
    MTL::ComputePipelineState* pPcgPrecondDot = nullptr;
    MTL::ComputePipelineState* pPcgUpdateP = nullptr;
    MTL::ComputePipelineState* pPcgUpdateSolRes = nullptr;
    MTL::ComputePipelineState* pDotPartial = nullptr;
    MTL::ComputePipelineState* pReduceScalars = nullptr;
    MTL::ComputePipelineState* pBicgUpdateP = nullptr;
    MTL::ComputePipelineState* pBicgSRes = nullptr;
    MTL::ComputePipelineState* pBicgEarlyUpdate = nullptr;
    MTL::ComputePipelineState* pBicgUpdateSolRes = nullptr;

    bool good = false;
    std::string description = "metal4Foam: no device";

    // Topology
    uint32_t n = 0;        // cells (rows)
    uint32_t nf = 0;       // internal faces
    uint32_t ngroups = 0;  // threadgroups for length-n dispatches
    const int32_t* lastLow = nullptr;
    const int32_t* lastUpp = nullptr;

    // Addressing (uint32, static per mesh)
    MTL::Buffer* bLow = nullptr;
    MTL::Buffer* bUpp = nullptr;
    MTL::Buffer* bOwnStart = nullptr;
    MTL::Buffer* bLosort = nullptr;
    MTL::Buffer* bLosortStart = nullptr;

    // Coefficients (FP32, updated per solve)
    MTL::Buffer* bDiag = nullptr;
    MTL::Buffer* bUpper = nullptr;
    MTL::Buffer* bLower = nullptr;   // == bUpper for symmetric matrices
    bool ownLower = false;
    MTL::Buffer* bRD = nullptr;      // 1/diag

    // Work vectors (FP32)
    MTL::Buffer* bRA = nullptr;
    MTL::Buffer* bP = nullptr;
    MTL::Buffer* bWA = nullptr;
    MTL::Buffer* bDelta = nullptr;
    MTL::Buffer* bDeltaBest = nullptr;
    MTL::Buffer* bChebW = nullptr;
    MTL::Buffer* bChebV = nullptr;
    // BiCGStab extras
    MTL::Buffer* bRA0 = nullptr;
    MTL::Buffer* bYA = nullptr;
    MTL::Buffer* bAYA = nullptr;
    MTL::Buffer* bSA = nullptr;
    MTL::Buffer* bZA = nullptr;
    MTL::Buffer* bTA = nullptr;

    MTL::Buffer* bP1 = nullptr;
    MTL::Buffer* bP2 = nullptr;
    MTL::Buffer* bP3 = nullptr;
    MTL::Buffer* bScalars = nullptr;

    ~Impl()
    {
        auto rel = [](auto*& p) { if (p) { p->release(); p = nullptr; } };
        rel(pSpmvDotw); rel(pSpmv); rel(pChebInit); rel(pChebStep); rel(pEmul); rel(pPcgPrecondDot); rel(pPcgUpdateP);
        rel(pPcgUpdateSolRes); rel(pDotPartial); rel(pReduceScalars);
        rel(pBicgUpdateP); rel(pBicgSRes); rel(pBicgEarlyUpdate);
        rel(pBicgUpdateSolRes);
        releaseTopology();
        releaseVectors();
        rel(library); rel(queue); rel(device);
    }

    void releaseTopology()
    {
        auto rel = [](MTL::Buffer*& p) { if (p) { p->release(); p = nullptr; } };
        rel(bLow); rel(bUpp); rel(bOwnStart); rel(bLosort); rel(bLosortStart);
    }

    void releaseVectors()
    {
        auto rel = [](MTL::Buffer*& p) { if (p) { p->release(); p = nullptr; } };
        rel(bDiag); rel(bUpper);
        if (ownLower) rel(bLower); else bLower = nullptr;
        rel(bRD);
        rel(bRA); rel(bP); rel(bWA); rel(bDelta); rel(bDeltaBest);
        rel(bChebW); rel(bChebV);
        rel(bRA0); rel(bYA); rel(bAYA); rel(bSA); rel(bZA); rel(bTA);
        rel(bP1); rel(bP2); rel(bP3); rel(bScalars);
    }

    MTL::Buffer* newShared(size_t bytes)
    {
        return device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    }

    MTL::ComputePipelineState* makePipeline(const char* name, std::string& err)
    {
        NS::String* fname =
            NS::String::string(name, NS::UTF8StringEncoding);
        MTL::Function* fn = library->newFunction(fname);
        if (!fn)
        {
            err += std::string("missing kernel: ") + name + "; ";
            return nullptr;
        }
        NS::Error* nserr = nullptr;
        MTL::ComputePipelineState* pso =
            device->newComputePipelineState(fn, &nserr);
        fn->release();
        if (!pso)
        {
            err += std::string("pipeline failed: ") + name;
            if (nserr && nserr->localizedDescription())
            {
                err += ": ";
                err += nserr->localizedDescription()->utf8String();
            }
            err += "; ";
        }
        return pso;
    }

    void init()
    {
        device = MTL::CreateSystemDefaultDevice();
        if (!device)
        {
            description = "metal4Foam: no Metal device available";
            return;
        }
        queue = device->newCommandQueue();

        // Prefer a precompiled metallib; fall back to runtime compilation
        // of the embedded MSL source (no offline toolchain required).
        std::string libSource = "embedded-source (runtime compiled)";
        {
            std::string path;
            if (const char* p = std::getenv("METAL4FOAM_METALLIB"))
            {
                path = p;
            }
            else if (const char* p = std::getenv("FOAM_USER_LIBBIN"))
            {
                path = std::string(p) + "/metalFoamKernels.metallib";
            }

            if (!path.empty())
            {
                FILE* f = std::fopen(path.c_str(), "rb");
                if (f)
                {
                    std::fclose(f);
                    NS::Error* nserr = nullptr;
                    NS::String* nsPath = NS::String::string
                    (
                        path.c_str(), NS::UTF8StringEncoding
                    );
                    library = device->newLibrary(nsPath, &nserr);
                    if (library) libSource = path;
                }
            }
        }

        if (!library)
        {
            NS::Error* nserr = nullptr;
            MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
            NS::String* src = NS::String::string
            (
                Foam::metalFoamKernelsMSL, NS::UTF8StringEncoding
            );
            library = device->newLibrary(src, opts, &nserr);
            opts->release();
            if (!library)
            {
                description = "metal4Foam: MSL compile failed";
                if (nserr && nserr->localizedDescription())
                {
                    description += ": ";
                    description +=
                        nserr->localizedDescription()->utf8String();
                }
                return;
            }
        }

        std::string err;
        pSpmvDotw         = makePipeline("ldu_spmv_dotw", err);
        pSpmv             = makePipeline("ldu_spmv", err);
        pChebInit         = makePipeline("cheb_init", err);
        pChebStep         = makePipeline("cheb_step", err);
        pEmul             = makePipeline("emul", err);
        pPcgPrecondDot    = makePipeline("pcg_precond_dot", err);
        pPcgUpdateP       = makePipeline("pcg_update_p", err);
        pPcgUpdateSolRes  = makePipeline("pcg_update_sol_res", err);
        pDotPartial       = makePipeline("dot_partial", err);
        pReduceScalars    = makePipeline("reduce_scalars", err);
        pBicgUpdateP      = makePipeline("bicg_update_p", err);
        pBicgSRes         = makePipeline("bicg_s_res", err);
        pBicgEarlyUpdate  = makePipeline("bicg_early_update", err);
        pBicgUpdateSolRes = makePipeline("bicg_update_sol_res", err);

        if (!err.empty())
        {
            description = "metal4Foam: " + err;
            return;
        }

        good = true;
        description = "metal4Foam: ";
        description += device->name()->utf8String();
        description += " [" + libSource + "]";
    }

    // ---- encode helpers ------------------------------------------------

    void dispatchN(MTL::ComputeCommandEncoder* enc, uint32_t count)
    {
        const uint32_t g = (count + TG - 1)/TG;
        enc->dispatchThreadgroups
        (
            MTL::Size(g, 1, 1), MTL::Size(TG, 1, 1)
        );
    }

    void encSpmvDotw
    (
        MTL::ComputeCommandEncoder* enc,
        MTL::Buffer* x, MTL::Buffer* w, MTL::Buffer* y, MTL::Buffer* partials
    )
    {
        enc->setComputePipelineState(pSpmvDotw);
        enc->setBuffer(bDiag, 0, 0);
        enc->setBuffer(bUpper, 0, 1);
        enc->setBuffer(bLower, 0, 2);
        enc->setBuffer(bLow, 0, 3);
        enc->setBuffer(bUpp, 0, 4);
        enc->setBuffer(bOwnStart, 0, 5);
        enc->setBuffer(bLosort, 0, 6);
        enc->setBuffer(bLosortStart, 0, 7);
        enc->setBuffer(x, 0, 8);
        enc->setBuffer(w, 0, 9);
        enc->setBuffer(y, 0, 10);
        enc->setBuffer(partials, 0, 11);
        enc->setBytes(&n, sizeof(n), 12);
        dispatchN(enc, n);
    }

    void encSpmv
    (
        MTL::ComputeCommandEncoder* enc,
        MTL::Buffer* x, MTL::Buffer* y
    )
    {
        enc->setComputePipelineState(pSpmv);
        enc->setBuffer(bDiag, 0, 0);
        enc->setBuffer(bUpper, 0, 1);
        enc->setBuffer(bLower, 0, 2);
        enc->setBuffer(bLow, 0, 3);
        enc->setBuffer(bUpp, 0, 4);
        enc->setBuffer(bOwnStart, 0, 5);
        enc->setBuffer(bLosort, 0, 6);
        enc->setBuffer(bLosortStart, 0, 7);
        enc->setBuffer(x, 0, 8);
        enc->setBuffer(y, 0, 9);
        enc->setBytes(&n, sizeof(n), 10);
        dispatchN(enc, n);
    }

    // Chebyshev polynomial preconditioner: z ~= A^-1 r on spectrum
    // [beta/ratio, beta] of rD*A; all recurrence constants are known at
    // encode time, so the whole apply lands in the same command buffer.
    void encCheb
    (
        MTL::ComputeCommandEncoder* enc,
        MTL::Buffer* r, MTL::Buffer* z,
        int order, double beta, double ratio
    )
    {
        const double alpha = beta/ratio;
        const double theta = 0.5*(beta + alpha);
        const double delta = 0.5*(beta - alpha);
        const double sigma = theta/delta;
        double rho = 1.0/sigma;

        const float c1 = float(1.0/theta);
        enc->setComputePipelineState(pChebInit);
        enc->setBuffer(r, 0, 0);
        enc->setBuffer(bRD, 0, 1);
        enc->setBuffer(bChebW, 0, 2);
        enc->setBuffer(z, 0, 3);
        enc->setBytes(&c1, sizeof(c1), 4);
        enc->setBytes(&n, sizeof(n), 5);
        dispatchN(enc, n);

        for (int i = 1; i < order; ++i)
        {
            encSpmv(enc, z, bChebV);

            const double rhoNew = 1.0/(2.0*sigma - rho);
            const float c2 = float(rhoNew*rho);
            const float c3 = float(2.0*rhoNew/delta);
            rho = rhoNew;

            enc->setComputePipelineState(pChebStep);
            enc->setBuffer(r, 0, 0);
            enc->setBuffer(bChebV, 0, 1);
            enc->setBuffer(bRD, 0, 2);
            enc->setBuffer(bChebW, 0, 3);
            enc->setBuffer(z, 0, 4);
            enc->setBytes(&c2, sizeof(c2), 5);
            enc->setBytes(&c3, sizeof(c3), 6);
            enc->setBytes(&n, sizeof(n), 7);
            dispatchN(enc, n);
        }
    }

    void encEmul
    (
        MTL::ComputeCommandEncoder* enc,
        MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* out
    )
    {
        enc->setComputePipelineState(pEmul);
        enc->setBuffer(a, 0, 0);
        enc->setBuffer(b, 0, 1);
        enc->setBuffer(out, 0, 2);
        enc->setBytes(&n, sizeof(n), 3);
        dispatchN(enc, n);
    }

    void encReduce
    (
        MTL::ComputeCommandEncoder* enc,
        MTL::Buffer* P1, MTL::Buffer* P2,
        uint32_t mode, uint32_t firstIter
    )
    {
        enc->setComputePipelineState(pReduceScalars);
        enc->setBuffer(P1, 0, 0);
        enc->setBuffer(P2, 0, 1);
        enc->setBuffer(bScalars, 0, 2);
        enc->setBytes(&ngroups, sizeof(ngroups), 3);
        enc->setBytes(&mode, sizeof(mode), 4);
        enc->setBytes(&firstIter, sizeof(firstIter), 5);
        enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TG, 1, 1));
    }
};


// * * * * * * * * * * * * * * public interface  * * * * * * * * * * * * * //

Foam::metalFoamBackend::metalFoamBackend()
:
    impl_(new Impl)
{
    FpTrapGuard fpe;
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    impl_->init();
    pool->release();
}


Foam::metalFoamBackend::~metalFoamBackend() = default;


bool Foam::metalFoamBackend::good() const
{
    return impl_->good;
}


const char* Foam::metalFoamBackend::description() const
{
    return impl_->description.c_str();
}


const float* Foam::metalFoamBackend::delta() const
{
    return static_cast<const float*>(impl_->bDelta->contents());
}


void Foam::metalFoamBackend::setTopology
(
    int32_t nCells,
    int32_t nFaces,
    const int32_t* lowAddr,
    const int32_t* uppAddr,
    const int32_t* ownStart,
    const int32_t* losort,
    const int32_t* losortStart
)
{
    Impl& d = *impl_;

    FpTrapGuard fpe;

    const bool unchanged =
    (
        d.n == uint32_t(nCells)
     && d.nf == uint32_t(nFaces)
     && d.lastLow == lowAddr
     && d.lastUpp == uppAddr
     && d.bLow
    );
    if (unchanged) return;

    d.releaseTopology();
    d.releaseVectors();

    d.n = uint32_t(nCells);
    d.nf = uint32_t(nFaces);
    d.ngroups = (d.n + TG - 1)/TG;
    d.lastLow = lowAddr;
    d.lastUpp = uppAddr;

    // OpenFOAM labels are non-negative int32 here: bit-identical to uint32
    auto upload = [&](const int32_t* src, size_t count) -> MTL::Buffer*
    {
        MTL::Buffer* b = d.newShared(count*sizeof(uint32_t));
        std::memcpy(b->contents(), src, count*sizeof(uint32_t));
        return b;
    };

    d.bLow = upload(lowAddr, nFaces);
    d.bUpp = upload(uppAddr, nFaces);
    d.bOwnStart = upload(ownStart, size_t(nCells) + 1);
    d.bLosort = upload(losort, nFaces);
    d.bLosortStart = upload(losortStart, size_t(nCells) + 1);

    // Coefficient and work buffers
    d.bDiag = d.newShared(d.n*sizeof(float));
    d.bUpper = d.newShared(d.nf*sizeof(float));
    d.bLower = nullptr;
    d.ownLower = false;
    d.bRD = d.newShared(d.n*sizeof(float));

    d.bRA = d.newShared(d.n*sizeof(float));
    d.bP = d.newShared(d.n*sizeof(float));
    d.bWA = d.newShared(d.n*sizeof(float));
    d.bDelta = d.newShared(d.n*sizeof(float));
    d.bDeltaBest = d.newShared(d.n*sizeof(float));

    d.bP1 = d.newShared(d.ngroups*sizeof(float));
    d.bP2 = d.newShared(d.ngroups*sizeof(float));
    d.bP3 = d.newShared(d.ngroups*sizeof(float));
    d.bScalars = d.newShared(S_COUNT*sizeof(float));
}


double Foam::metalFoamBackend::setCoeffs
(
    const double* diag,
    const double* upper,
    const double* lower
)
{
    Impl& d = *impl_;
    FpTrapGuard fpe;
    const double t0 = nowSeconds();

    // FP64 -> FP32 conversions at memory bandwidth (unified memory:
    // this is the entire "host->device transfer")
    vDSP_vdpsp(diag, 1, static_cast<float*>(d.bDiag->contents()), 1, d.n);
    vDSP_vdpsp(upper, 1, static_cast<float*>(d.bUpper->contents()), 1, d.nf);

    if (lower)
    {
        if (!d.ownLower)
        {
            d.bLower = d.newShared(d.nf*sizeof(float));
            d.ownLower = true;
        }
        vDSP_vdpsp(lower, 1, static_cast<float*>(d.bLower->contents()), 1, d.nf);
    }
    else
    {
        if (d.ownLower && d.bLower)
        {
            d.bLower->release();
            d.ownLower = false;
        }
        d.bLower = d.bUpper;   // symmetric
    }

    // Jacobi preconditioner: 1/diag
    float* rd = static_cast<float*>(d.bRD->contents());
    for (uint32_t i = 0; i < d.n; ++i)
    {
        rd[i] = float(1.0/diag[i]);
    }

    return nowSeconds() - t0;
}


Foam::metalFoamBackend::Stats Foam::metalFoamBackend::pcg
(
    const double* r,
    double targetResSum,
    int maxIter,
    int iterationsPerSync,
    int chebOrder,
    double chebBeta,
    double chebRatio
)
{
    Impl& d = *impl_;
    FpTrapGuard fpe;
    Stats stats;
    if (!d.good || d.n == 0) { stats.singular = true; return stats; }

    if (chebOrder > 1 && !d.bChebW)
    {
        d.bChebW = d.newShared(d.n*sizeof(float));
        d.bChebV = d.newShared(d.n*sizeof(float));
    }

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    const double tw0 = nowSeconds();

    vDSP_vdpsp(r, 1, static_cast<float*>(d.bRA->contents()), 1, d.n);
    std::memset(d.bDelta->contents(), 0, d.n*sizeof(float));
    std::memset(d.bDeltaBest->contents(), 0, d.n*sizeof(float));
    std::memset(d.bP->contents(), 0, d.n*sizeof(float));
    std::memset(d.bScalars->contents(), 0, S_COUNT*sizeof(float));

    const float* s = static_cast<const float*>(d.bScalars->contents());

    if (iterationsPerSync < 1) iterationsPerSync = 1;

    int it = 0;
    double bestRes = 1.0e300;
    int noProgress = 0;
    constexpr int stagLimit = 50;

    while (it < maxIter)
    {
        const int batch =
            std::min(iterationsPerSync, maxIter - it);

        MTL::CommandBuffer* cmd = d.queue->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

        for (int k = 0; k < batch; ++k)
        {
            const uint32_t firstIter = (it + k == 0) ? 1u : 0u;

            if (chebOrder > 1)
            {
                // wA = Cheb(A, rA) ; P1 partials of wA.rA
                d.encCheb(enc, d.bRA, d.bWA, chebOrder, chebBeta, chebRatio);
                enc->setComputePipelineState(d.pDotPartial);
                enc->setBuffer(d.bWA, 0, 0);
                enc->setBuffer(d.bRA, 0, 1);
                enc->setBuffer(d.bP1, 0, 2);
                enc->setBytes(&d.n, sizeof(d.n), 3);
                d.dispatchN(enc, d.n);
            }
            else
            {
                // wA = rD*rA ; P1 partials of wA.rA
                enc->setComputePipelineState(d.pPcgPrecondDot);
                enc->setBuffer(d.bRA, 0, 0);
                enc->setBuffer(d.bRD, 0, 1);
                enc->setBuffer(d.bWA, 0, 2);
                enc->setBuffer(d.bP1, 0, 3);
                enc->setBytes(&d.n, sizeof(d.n), 4);
                d.dispatchN(enc, d.n);
            }

            d.encReduce(enc, d.bP1, d.bP1, 0, firstIter);

            // p = wA + beta*p
            enc->setComputePipelineState(d.pPcgUpdateP);
            enc->setBuffer(d.bP, 0, 0);
            enc->setBuffer(d.bWA, 0, 1);
            enc->setBuffer(d.bScalars, 0, 2);
            enc->setBytes(&d.n, sizeof(d.n), 3);
            d.dispatchN(enc, d.n);

            // wA = A*p ; P2 partials of p.(A*p)
            d.encSpmvDotw(enc, d.bP, d.bP, d.bWA, d.bP2);
            d.encReduce(enc, d.bP2, d.bP2, 1, 0);

            // delta += alpha*p ; rA -= alpha*wA ; P3 partials of |rA|
            enc->setComputePipelineState(d.pPcgUpdateSolRes);
            enc->setBuffer(d.bDelta, 0, 0);
            enc->setBuffer(d.bRA, 0, 1);
            enc->setBuffer(d.bP, 0, 2);
            enc->setBuffer(d.bWA, 0, 3);
            enc->setBuffer(d.bScalars, 0, 4);
            enc->setBuffer(d.bP3, 0, 5);
            enc->setBytes(&d.n, sizeof(d.n), 6);
            d.dispatchN(enc, d.n);

            d.encReduce(enc, d.bP3, d.bP3, 2, 0);
        }

        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        stats.tGpuSeconds += cmd->GPUEndTime() - cmd->GPUStartTime();

        it += batch;
        stats.nIterations = it;
        const int batchArg = batch;

        const double res = double(s[S_RES]);
        stats.finalResSum = res;

        if (!std::isfinite(res) || !std::isfinite(double(s[S_WAPA])))
        {
            stats.singular = true;
            break;
        }
        // Track the minimum-residual iterate: FP32 CG on ill-conditioned
        // systems oscillates near its precision floor, and iterative
        // refinement is only sound if each inner solve returns an
        // iterate no worse than its start.
        if (res < bestRes)
        {
            bestRes = res;
            noProgress = 0;
            std::memcpy
            (
                d.bDeltaBest->contents(), d.bDelta->contents(),
                d.n*sizeof(float)
            );
        }
        else
        {
            noProgress += batchArg;
        }
        if (res <= targetResSum)
        {
            stats.converged = true;
            break;
        }
        if (noProgress >= stagLimit)
        {
            stats.stagnated = true;
            break;
        }
    }

    // Return the best iterate seen, not necessarily the last one
    if (bestRes < stats.finalResSum)
    {
        std::memcpy
        (
            d.bDelta->contents(), d.bDeltaBest->contents(),
            d.n*sizeof(float)
        );
        stats.finalResSum = bestRes;
    }

    stats.tWallSeconds = nowSeconds() - tw0;
    pool->release();
    return stats;
}


Foam::metalFoamBackend::Stats Foam::metalFoamBackend::bicgstab
(
    const double* r,
    double targetResSum,
    int maxIter
)
{
    Impl& d = *impl_;
    FpTrapGuard fpe;
    Stats stats;
    if (!d.good || d.n == 0) { stats.singular = true; return stats; }

    // Lazily allocate the BiCGStab-only vectors
    if (!d.bRA0)
    {
        d.bRA0 = d.newShared(d.n*sizeof(float));
        d.bYA = d.newShared(d.n*sizeof(float));
        d.bAYA = d.newShared(d.n*sizeof(float));
        d.bSA = d.newShared(d.n*sizeof(float));
        d.bZA = d.newShared(d.n*sizeof(float));
        d.bTA = d.newShared(d.n*sizeof(float));
    }

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    const double tw0 = nowSeconds();

    vDSP_vdpsp(r, 1, static_cast<float*>(d.bRA->contents()), 1, d.n);
    std::memcpy
    (
        d.bRA0->contents(), d.bRA->contents(), d.n*sizeof(float)
    );
    std::memset(d.bDelta->contents(), 0, d.n*sizeof(float));
    std::memset(d.bDeltaBest->contents(), 0, d.n*sizeof(float));
    std::memset(d.bP->contents(), 0, d.n*sizeof(float));
    std::memset(d.bAYA->contents(), 0, d.n*sizeof(float));
    std::memset(d.bScalars->contents(), 0, S_COUNT*sizeof(float));

    const float* s = static_cast<const float*>(d.bScalars->contents());

    int it = 0;
    double bestRes = 1.0e300;
    int noProgress = 0;
    constexpr int stagLimit = 50;

    while (it < maxIter)
    {
        const uint32_t firstIter = (it == 0) ? 1u : 0u;

        // ---- first half: p, yA = M^-1 p, AyA = A yA, alpha, sA ----------
        {
            MTL::CommandBuffer* cmd = d.queue->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

            // P1 partials of rA0.rA ; then rA0rA, beta
            enc->setComputePipelineState(d.pDotPartial);
            enc->setBuffer(d.bRA0, 0, 0);
            enc->setBuffer(d.bRA, 0, 1);
            enc->setBuffer(d.bP1, 0, 2);
            enc->setBytes(&d.n, sizeof(d.n), 3);
            d.dispatchN(enc, d.n);
            d.encReduce(enc, d.bP1, d.bP1, 3, firstIter);

            // p = rA + beta*(p - omega*AyA)
            enc->setComputePipelineState(d.pBicgUpdateP);
            enc->setBuffer(d.bP, 0, 0);
            enc->setBuffer(d.bRA, 0, 1);
            enc->setBuffer(d.bAYA, 0, 2);
            enc->setBuffer(d.bScalars, 0, 3);
            enc->setBytes(&d.n, sizeof(d.n), 4);
            d.dispatchN(enc, d.n);

            // yA = rD*p ; AyA = A*yA with dot(rA0, AyA) -> P2
            d.encEmul(enc, d.bRD, d.bP, d.bYA);
            d.encSpmvDotw(enc, d.bYA, d.bRA0, d.bAYA, d.bP2);
            d.encReduce(enc, d.bP2, d.bP2, 4, 0);

            // sA = rA - alpha*AyA ; P3 partials of |sA|
            enc->setComputePipelineState(d.pBicgSRes);
            enc->setBuffer(d.bSA, 0, 0);
            enc->setBuffer(d.bRA, 0, 1);
            enc->setBuffer(d.bAYA, 0, 2);
            enc->setBuffer(d.bScalars, 0, 3);
            enc->setBuffer(d.bP3, 0, 4);
            enc->setBytes(&d.n, sizeof(d.n), 5);
            d.dispatchN(enc, d.n);
            d.encReduce(enc, d.bP3, d.bP3, 2, 0);

            enc->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();
            stats.tGpuSeconds += cmd->GPUEndTime() - cmd->GPUStartTime();
        }

        // breakdown checks (post-hoc analogue of checkSingularity)
        if
        (
            !std::isfinite(double(s[S_RA0RA]))
         || !std::isfinite(double(s[S_ALPHA]))
         || !std::isfinite(double(s[S_RES]))
        )
        {
            stats.singular = true;
            break;
        }

        // sA convergence test (matches the built-in early exit)
        {
            const double res = double(s[S_RES]);
            if (res <= targetResSum)
            {
                MTL::CommandBuffer* cmd = d.queue->commandBuffer();
                MTL::ComputeCommandEncoder* enc =
                    cmd->computeCommandEncoder();
                enc->setComputePipelineState(d.pBicgEarlyUpdate);
                enc->setBuffer(d.bDelta, 0, 0);
                enc->setBuffer(d.bYA, 0, 1);
                enc->setBuffer(d.bScalars, 0, 2);
                enc->setBytes(&d.n, sizeof(d.n), 3);
                d.dispatchN(enc, d.n);
                enc->endEncoding();
                cmd->commit();
                cmd->waitUntilCompleted();
                stats.tGpuSeconds += cmd->GPUEndTime() - cmd->GPUStartTime();

                stats.nIterations = ++it;
                stats.finalResSum = res;
                stats.converged = true;
                stats.tWallSeconds = nowSeconds() - tw0;
                pool->release();
                return stats;
            }
        }

        // ---- second half: zA, tA, omega, solution + residual update -----
        {
            MTL::CommandBuffer* cmd = d.queue->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

            // zA = rD*sA ; tA = A*zA with dot(sA, tA) -> P2
            d.encEmul(enc, d.bRD, d.bSA, d.bZA);
            d.encSpmvDotw(enc, d.bZA, d.bSA, d.bTA, d.bP2);

            // P1 partials of tA.tA
            enc->setComputePipelineState(d.pDotPartial);
            enc->setBuffer(d.bTA, 0, 0);
            enc->setBuffer(d.bTA, 0, 1);
            enc->setBuffer(d.bP1, 0, 2);
            enc->setBytes(&d.n, sizeof(d.n), 3);
            d.dispatchN(enc, d.n);

            // omega = (tA.sA)/(tA.tA)
            d.encReduce(enc, d.bP1, d.bP2, 5, 0);

            // delta += alpha*yA + omega*zA ; rA = sA - omega*tA ; |rA| -> P3
            enc->setComputePipelineState(d.pBicgUpdateSolRes);
            enc->setBuffer(d.bDelta, 0, 0);
            enc->setBuffer(d.bYA, 0, 1);
            enc->setBuffer(d.bZA, 0, 2);
            enc->setBuffer(d.bRA, 0, 3);
            enc->setBuffer(d.bSA, 0, 4);
            enc->setBuffer(d.bTA, 0, 5);
            enc->setBuffer(d.bScalars, 0, 6);
            enc->setBuffer(d.bP3, 0, 7);
            enc->setBytes(&d.n, sizeof(d.n), 8);
            d.dispatchN(enc, d.n);
            d.encReduce(enc, d.bP3, d.bP3, 2, 0);

            enc->endEncoding();
            cmd->commit();
            cmd->waitUntilCompleted();
            stats.tGpuSeconds += cmd->GPUEndTime() - cmd->GPUStartTime();
        }

        ++it;
        stats.nIterations = it;
        const int batchArg = 1;

        const double res = double(s[S_RES]);
        stats.finalResSum = res;

        if (!std::isfinite(res) || !std::isfinite(double(s[S_OMEGA])))
        {
            stats.singular = true;
            break;
        }
        if (res < bestRes)
        {
            bestRes = res;
            noProgress = 0;
            std::memcpy
            (
                d.bDeltaBest->contents(), d.bDelta->contents(),
                d.n*sizeof(float)
            );
        }
        else
        {
            noProgress += batchArg;
        }
        if (res <= targetResSum)
        {
            stats.converged = true;
            break;
        }
        if (noProgress >= stagLimit)
        {
            stats.stagnated = true;
            break;
        }
    }

    if (bestRes < stats.finalResSum)
    {
        std::memcpy
        (
            d.bDelta->contents(), d.bDeltaBest->contents(),
            d.n*sizeof(float)
        );
        stats.finalResSum = bestRes;
    }

    stats.tWallSeconds = nowSeconds() - tw0;
    pool->release();
    return stats;
}

// ************************************************************************* //
