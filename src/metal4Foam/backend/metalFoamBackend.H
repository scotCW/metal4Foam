/*---------------------------------------------------------------------------*\
    metal4Foam: Metal-accelerated linear solvers for OpenFOAM (Apple GPU)

    Copyright (C) 2026 metal4Foam authors
    SPDX-License-Identifier: GPL-3.0-or-later

Class
    Foam::metalFoamBackend

Description
    GPU backend for FP32 Krylov iterations on Apple Silicon via Metal.

    The matrix stays in OpenFOAM's native LDU (face-addressed) layout on
    the GPU; because of unified memory the coefficient "upload" per solve
    is only a double->float precision conversion (vDSP) into shared
    MTLBuffers -- no PCIe-style transfer exists on this hardware.

    Metal has no FP64 in shaders, so both solvers run in FP32 on the
    *correction* system A*delta = r0 (delta0 = 0); the caller accumulates
    psi += delta in FP64 and may restart (iterative refinement) to reach
    tolerances beyond single-precision reach.

    This header exposes no Metal types (pimpl) so that OpenFOAM
    translation units never include metal-cpp.

\*---------------------------------------------------------------------------*/

#ifndef Foam_metalFoamBackend_H
#define Foam_metalFoamBackend_H

#include <memory>
#include <cstdint>

namespace Foam
{

class metalFoamBackend
{
public:

    struct Stats
    {
        int nIterations = 0;      //- inner FP32 iterations performed
        double finalResSum = 0;   //- FP32 sum|r| at exit (unnormalised)
        bool converged = false;   //- hit the requested threshold
        bool singular = false;    //- breakdown (NaN/Inf in scalars)
        bool stagnated = false;   //- no progress at FP32 precision
        double tGpuSeconds = 0;   //- accumulated GPU execution time
        double tWallSeconds = 0;  //- wall time inside the iteration loop
    };

    metalFoamBackend();
    ~metalFoamBackend();

    metalFoamBackend(const metalFoamBackend&) = delete;
    void operator=(const metalFoamBackend&) = delete;

    //- (Re)build addressing buffers. int32 label build assumed.
    //  Cheap no-op when (nCells,nFaces,low,upp) are unchanged.
    void setTopology
    (
        int32_t nCells,
        int32_t nFaces,
        const int32_t* lowAddr,
        const int32_t* uppAddr,
        const int32_t* ownStart,     // nCells+1
        const int32_t* losort,       // nFaces
        const int32_t* losortStart   // nCells+1
    );

    //- Convert and stage matrix coefficients (FP64 -> FP32).
    //  lower == nullptr selects the symmetric path (lower = upper).
    //  Returns seconds spent converting.
    double setCoeffs
    (
        const double* diag,
        const double* upper,
        const double* lower
    );

    //- FP32 PCG (Jacobi preconditioned) on A*delta = r.
    //  Stops when sum|r32| <= targetResSum, on stagnation, or maxIter.
    Stats pcg
    (
        const double* r,           // initial residual (nCells)
        double targetResSum,       // unnormalised L1 target
        int maxIter,
        int iterationsPerSync = 1,
        int chebOrder = 1,         // >1: Chebyshev-Jacobi preconditioner
        double chebBeta = 2.0,     // upper spectral bound of rD*A
        double chebRatio = 30.0    // covered spectrum [beta/ratio, beta]
    );

    //- FP32 BiCGStab (Jacobi preconditioned) on A*delta = r.
    Stats bicgstab
    (
        const double* r,
        double targetResSum,
        int maxIter
    );

    //- Correction vector from the last pcg()/bicgstab() call (shared mem)
    const float* delta() const;

    //- True when a Metal device and all pipelines are available
    bool good() const;

    //- Human-readable device/library description
    const char* description() const;

private:

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // End namespace Foam

#endif

// ************************************************************************* //
