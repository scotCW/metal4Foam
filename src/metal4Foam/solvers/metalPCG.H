/*---------------------------------------------------------------------------*\
    metal4Foam: Metal-accelerated linear solvers for OpenFOAM (Apple GPU)

    Copyright (C) 2026 metal4Foam authors
    SPDX-License-Identifier: GPL-3.0-or-later

Class
    Foam::metalPCG

Group
    grpLduMatrixSolvers

Description
    Preconditioned conjugate gradient solver for symmetric lduMatrices,
    with the Krylov iterations executed in FP32 on the Apple GPU (Metal)
    and FP64 iterative refinement on the CPU. Jacobi (diagonal)
    preconditioning.

    Selected in fvSolution exactly like the built-in solvers:

    \verbatim
    p
    {
        solver          metalPCG;
        tolerance       1e-06;
        relTol          0.05;
    }
    \endverbatim

SourceFiles
    metalPCG.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_metalPCG_H
#define Foam_metalPCG_H

#include "metalFoamSolver.H"

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class metalPCG Declaration
\*---------------------------------------------------------------------------*/

class metalPCG
:
    public metalFoamSolver
{
public:

    //- Runtime type information
    TypeName("metalPCG");


    // Constructors

        //- Construct from matrix components and solver controls
        metalPCG
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );


    //- Destructor
    virtual ~metalPCG() = default;


    // Member Functions

        //- Solve with given field and rhs (in solveScalar precision)
        virtual solverPerformance scalarSolve
        (
            solveScalarField& psi,
            const solveScalarField& source,
            const direction cmpt = 0
        ) const;

        //- Solve the matrix with this solver
        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt = 0
        ) const;
};

} // End namespace Foam

#endif

// ************************************************************************* //
