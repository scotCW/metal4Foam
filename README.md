# metal4Foam — Metal-accelerated linear solvers for OpenFOAM on Apple Silicon

A runtime-selectable OpenFOAM linear-solver plugin backed by Apple's Metal
API. Structurally modeled on
[petsc4Foam](https://develop.openfoam.com/modules/external-solver) (and the
amgx4Foam pattern): a separate library implementing `lduMatrix::solver`,
registered through OpenFOAM's run-time selection tables, selected by name
from `fvSolution` — **zero modifications to OpenFOAM core**.

```
p
{
    solver          metalPCG;        // or metalPBiCGStab
    tolerance       1e-06;
    relTol          0.05;

    metal            // optional
    {
        maxRestarts        4;     // FP64 iterative-refinement restarts
        iterationsPerSync  1;     // GPU iterations per CPU sync (PCG)
        profile            false; // per-solve timing breakdown
        verbose            false; // per-restart residual trace
    }
}
```

and in `system/controlDict`:

```
libs            (metalFoam);
```

## What is provided

| Solver | Matrix types | Algorithm |
|---|---|---|
| `metalPCG` | symmetric | Conjugate gradient, Chebyshev–Jacobi polynomial preconditioner, constant-mode deflation for quasi-singular (pressure) matrices, FP32 on GPU + FP64 iterative refinement |
| `metalPBiCGStab` | symmetric + asymmetric | BiCGStab, Jacobi preconditioner, FP32 on GPU + FP64 iterative refinement |

Both mirror the built-in `PCG`/`PBiCGStab` control flow (same L1 residual
norm, same `normFactor`, same convergence tests), so residuals printed in
the log are directly comparable with the CPU solvers.

## Design

### Unified memory: what it buys, honestly

Apple Silicon's CPU and GPU share physical RAM. The plugin allocates all
GPU buffers as `MTLResourceStorageModeShared`, so there is **no
PCIe-style host<->device copy anywhere** — the CUDA/HIP plugins
(petsc4Foam, amgx4Foam) must stage matrix coefficients and fields across
the PCIe bus every solve; here the "transfer" is an in-RAM conversion.

Two honest caveats:

1. **True zero-copy wrapping of OpenFOAM's arrays is not possible.**
   `newBufferWithBytesNoCopy` requires page-aligned, page-multiple
   allocations; OpenFOAM's `List<scalar>` storage is ordinary malloc
   memory. So data lands in shared buffers via one memcpy-speed pass.
2. **Metal has no FP64.** OpenFOAM's default build is double precision;
   MSL supports only float/half. The per-solve "conversion" is therefore
   a fused FP64->FP32 precision conversion (Accelerate/vDSP `vDSP_vdpsp`,
   runs at memory bandwidth). This is also why the solver runs iterative
   refinement (below).

### The matrix never leaves LDU format

OpenFOAM's `lduMatrix` stores coefficients as `diag`/`upper`/`lower`
arrays with face-based `owner`/`neighbour` addressing — not CSR. Instead
of converting to CSR every solve (the natural thing for cuSPARSE/PETSc,
and a real per-solve cost there), the SpMV kernel here consumes the LDU
layout directly, row-per-thread:

```
y[i] = diag[i]*x[i]
     + sum over owned faces  f in [ownerStart[i], ownerStart[i+1]) : upper[f]*x[upp[f]]
     + sum over losort faces k in [losortStart[i], losortStart[i+1)): lower[losort[k]]*x[low[losort[k]]]
```

`ownerStartAddr()`/`losortAddr()`/`losortStartAddr()` are maintained by
`lduAddressing` itself and are static per mesh, so the *entire* per-solve
matrix update is three contiguous vDSP conversions (diag, upper, lower) —
there is no scatter, no column sorting, no structure rebuild. The
addressing arrays are uploaded once and cached per (mesh, field).

### Why not MPS?

`MPSMatrixVectorMultiplication` is a *dense* GEMV — unusable for a sparse
CFD matrix (an N^2 dense operand at N ~ 10^6 is ~4 TB). MPS has no
general sparse CSR/LDU SpMV primitive, and its reduction primitives
(`MPSNNReduce*`) target image/NN tensor layouts, not flat FP32 vectors
feeding a Krylov update on the same command buffer. So the seven small
kernels the solvers need (SpMV fused with a dot product, preconditioner
apply fused with a dot product, vector updates fused with L1-norm
partials, and a scalar-finalize pass) are custom MSL. Dot products use
the standard two-stage `simd_sum` reduction (simdgroup partials ->
threadgroup partial per block -> single-threadgroup finalize that also
computes alpha/beta/omega **on the GPU**, so an iteration needs no CPU
round-trip for its scalars).

### Precision strategy: FP32 inner, FP64 outer (iterative refinement)

Each solve works on the correction system `A*delta = r` with `delta0 = 0`:

1. CPU (FP64): `r = b - A*psi`, `normFactor`, initial residual — exactly
   the built-in PCG code path.
2. GPU (FP32): Krylov iterations until the FP32 L1 residual hits the
   target, stagnates at FP32 precision, or hits maxIter. The backend
   tracks the **minimum-residual iterate** and returns that, so a
   restart can never make things worse.
3. CPU (FP64): `psi += delta`, recompute the true residual, re-test
   convergence with OpenFOAM's own criteria; restart the GPU solve on the
   fresh residual if needed (up to `maxRestarts`).

Reported initial/final residuals are genuine FP64 quantities.

### The quasi-singular pressure matrix (the hard part)

The incompressible pressure equation is a Neumann Poisson problem: the
matrix is singular up to `setReference`, which pins one cell. That
regularisation creates one *weakly held* constant mode whose effective
eigenvalue shrinks with cell count — plain FP32 CG keeps re-exciting it
through roundoff and **stalls at a residual floor that grows with mesh
size** (measured: ~3e-3 relative at 100^2, ~3e-2 at 400^2), and
refinement restarts do not compose because each restart stalls the same
way. Three measures (all measured, see the report) fix this:

1. **Constant-mode deflation.** The solver computes row sums
   `u = A*1` (FP64). When almost all rows have `u ~ 0` (auto-detected;
   `metal { deflate on|off|auto; }`), the GPU receives the *exactly
   singular* operator (`diag0 = diag - u`), the inner RHS is projected
   to zero mean, and the constant mode is corrected outside in FP64:
   `psi += c` with `c = (u.r)/(u.u)` — an exact one-dimensional solve
   in the pinned direction.
2. **Chebyshev–Jacobi polynomial preconditioning** (`chebyshevOrder`,
   default 8; 1 = plain Jacobi). All recurrence constants are known at
   encode time, so the whole preconditioner application lands in the
   same command buffer: iterations drop ~8x with zero extra syncs,
   which also limits FP32 orthogonality loss. The spectral bound comes
   from a Gershgorin pass (sign-aware: `fvm::laplacian` produces
   negative diagonals).
3. **Hybrid polish.** The deflated operator cannot zero the
   reference-cell defect it excludes, so when a deflated restart stops
   progressing the solver hands the GPU the exact operator for the
   final approach to tolerance.

### Command-buffer / sync structure

One command buffer per iteration (PCG can batch `iterationsPerSync`),
with all scalar dependencies (beta, alpha, omega) resolved by tiny
on-GPU finalize kernels; the CPU reads back only the residual after
`waitUntilCompleted`. That single sync per iteration (~0.1–0.3 ms) is
the dominant cost at small mesh sizes — see the honest crossover
analysis in the benchmark report.

BiCGStab needs two syncs per iteration to reproduce the built-in
solver's mid-iteration early-exit test on `|sA|`.

### FPE-trap coexistence

OpenFOAM traps SIGFPE (`FOAM_SIGFPE`, `-ftrapping-math`); Apple's AGX
driver raises FP exceptions internally during device init and command
submission (this crashes a naive integration). Every backend entry point
installs an RAII guard that swaps in the default FP environment and
restores the caller's on exit (per-thread state), so OpenFOAM's trapping
still protects the CPU-side solver math.

## Build

Requirements: an OpenFOAM-v25xx arm64 Darwin build (`WM_PRECISION_OPTION=DP`,
`WM_LABEL_SIZE=32`), Xcode command-line tools. The offline Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`) is optional — without it
the kernels are runtime-compiled from embedded source.

```
source <OpenFOAM>/etc/bashrc
./Allwmake
```

`Allwmake` (patterned on petsc4Foam's):
1. embeds `shaders/metalFoamKernels.metal` into a generated header
   (runtime-compile fallback),
2. pre-compiles it to `$FOAM_USER_LIBBIN/metalFoamKernels.metallib` via
   `xcrun -sdk macosx metal` + `metallib` when the toolchain exists
   (wmake cannot do this natively), and
3. `wmake libso` -> `$FOAM_USER_LIBBIN/libmetalFoam.dylib`.

metal-cpp (Apple's official C++ Metal bindings) is vendored under
`thirdParty/metal-cpp`; the plugin contains **no Objective-C++** — the
backend is a single plain C++17 translation unit, and no OpenFOAM
translation unit sees a Metal header (pimpl).

## Measured performance (Apple M3 Pro, 36 GB)

Pressure solve per call, cavity, vs built-in `PCG`+DIC. Full numbers and
methodology in [BENCHMARK.md](BENCHMARK.md).

| Cells | CPU PCG+DIC | metalPCG | |
|---:|---:|---:|---|
| 400 | 0.066 ms | 1.109 ms | 16.8x **slower** |
| 4,900 | 6.43 ms | 8.33 ms | 1.3x **slower** |
| 10,000 | 20.5 ms | 11.6 ms | **1.76x faster** |
| 40,000 | 173.8 ms | 34.4 ms | **5.05x faster** |
| 160,000 | 1500.8 ms | 249.7 ms | **6.01x faster** |
| 640,000 | 10508.3 ms | 1787.8 ms | **5.88x faster** |
| 1,440,000 | *no convergence* | *no convergence* | — |

**Crossover ~7,000 cells** — below that the per-iteration CPU sync
(~0.1–0.3 ms) dominates and the CPU wins outright. Speedup saturates near
6x: the problem is bandwidth-bound and both processors share one memory
bus. Momentum (asymmetric) solves are **1.23x slower** on GPU — they
converge in ~15 iterations, so sync overhead dominates; offloading them is
not worthwhile.

## Scope and limitations

- **Serial only.** Coupled interfaces (processor/cyclic/AMI) fatal-error
  with a clear message. petsc4Foam bakes processor couplings into
  off-diagonal matrix entries; the same approach would extend here, but
  multi-GPU/MPI is out of scope.
- **Jacobi preconditioning only.** DIC/DILU are sequential triangular
  sweeps that do not map to GPU threads; expect ~1.5–2x the iteration
  count of DIC-preconditioned CPU PCG per solve.
- **No GAMG equivalent** (deliberately out of scope — AMG on Metal is a
  separate multi-month project). For cases where only multigrid
  converges acceptably, use CPU GAMG.
- **Not validated for topology-changing meshes** (addressing cache keys
  on array identity; a topo change triggers rebuild, but this path is
  untested).
- **FP32 precision floor at ~1.4M cells.** At 1200^2 the inner solve
  decouples from the true residual (GPU reports ~3e-6 while the FP64
  residual is ~1.2) because the converged per-cell residual is ~1e-13
  while individual SpMV terms are ~1e-6 — recovering that needs more than
  FP32's 7 digits. Outer refinement cannot rescue it. Note CPU PCG/DIC
  also fails to converge at this mesh size; the real answer there is GAMG.
  At 640k cells some intermediate solves under-converge slightly
  (continuity error still 2.9e-11), so that is the practical edge.

## Repository layout

```
metal4Foam/
├── Allwmake                          # embeds MSL, builds metallib + libso
├── src/metal4Foam/
│   ├── Make/{files,options}          # wmake integration (petsc4Foam pattern)
│   ├── solvers/metalFoamSolver.[CH]  # FP64 wrapper + refinement driver
│   ├── solvers/metalPCG.[CH]         # symmetric solver registration
│   ├── solvers/metalPBiCGStab.[CH]   # asymmetric solver registration
│   ├── backend/metalFoamBackend.[CH] # all Metal code (metal-cpp, pimpl)
│   └── shaders/metalFoamKernels.metal
├── thirdParty/metal-cpp/             # vendored Apple metal-cpp
└── benchmark/                        # cavity validation + benchmark harness
```
