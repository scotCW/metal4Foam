/*---------------------------------------------------------------------------*\
    metal4Foam: Metal-accelerated linear solvers for OpenFOAM (Apple GPU)

    Copyright (C) 2026 metal4Foam authors
    SPDX-License-Identifier: GPL-3.0-or-later

Class
    Foam::metalFoamSolver

Description
    Common driver for the Metal-backed lduMatrix solvers (metalPCG,
    metalPBiCGStab).

    The FP64 parts (initial residual, normalisation factor, psi updates,
    convergence bookkeeping) exactly mirror the built-in PCG/PBiCGStab;
    the Krylov iterations run in FP32 on the GPU against the correction
    system A*delta = r. Optional outer iterative refinement (default on)
    restarts the FP32 solve on the recomputed FP64 residual, so reported
    initial/final residuals are genuine double-precision quantities and
    directly comparable with the CPU solvers.

    Controls (all optional), inside the solver dict in fvSolution:

    \verbatim
    p
    {
        solver          metalPCG;
        tolerance       1e-06;
        relTol          0.05;

        metal
        {
            maxRestarts        4;    // outer FP32 refinement restarts
            iterationsPerSync  1;    // GPU iterations per CPU sync (PCG)
            profile            false;
            verbose            false;
        }
    }
    \endverbatim

SourceFiles
    metalFoamSolver.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_metalFoamSolver_H
#define Foam_metalFoamSolver_H

#include "lduMatrix.H"

namespace Foam
{

class metalFoamBackend;

/*---------------------------------------------------------------------------*\
                      Class metalFoamSolver Declaration
\*---------------------------------------------------------------------------*/

class metalFoamSolver
:
    public lduMatrix::solver
{
protected:

    // Protected Data

        //- Optional "metal" sub-dictionary controls
        dictionary metalDict_;


    // Protected Member Functions

        //- Get (or create) the cached backend for this mesh/field
        metalFoamBackend& backend() const;

        //- Common FP64 wrapper + FP32 GPU solve
        solverPerformance solveImpl
        (
            const word& solverName,
            const bool useBiCGStab,
            solveScalarField& psi,
            const solveScalarField& source,
            const direction cmpt
        ) const;

        //- No copy construct/assignment
        metalFoamSolver(const metalFoamSolver&) = delete;
        void operator=(const metalFoamSolver&) = delete;


public:

    // Constructors

        metalFoamSolver
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

    //- Destructor
    virtual ~metalFoamSolver() = default;
};

} // End namespace Foam

#endif

// ************************************************************************* //
