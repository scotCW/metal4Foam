/*---------------------------------------------------------------------------*\
    metal4Foam: Metal-accelerated linear solvers for OpenFOAM (Apple GPU)

    Copyright (C) 2026 metal4Foam authors
    SPDX-License-Identifier: GPL-3.0-or-later

Class
    Foam::metalPBiCGStab

Group
    grpLduMatrixSolvers

Description
    Preconditioned bi-conjugate gradient stabilised solver for symmetric
    and asymmetric lduMatrices, with the Krylov iterations executed in
    FP32 on the Apple GPU (Metal) and FP64 iterative refinement on the
    CPU. Jacobi (diagonal) preconditioning.

SourceFiles
    metalPBiCGStab.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_metalPBiCGStab_H
#define Foam_metalPBiCGStab_H

#include "metalFoamSolver.H"

namespace Foam
{

/*---------------------------------------------------------------------------*\
                        Class metalPBiCGStab Declaration
\*---------------------------------------------------------------------------*/

class metalPBiCGStab
:
    public metalFoamSolver
{
public:

    //- Runtime type information
    TypeName("metalPBiCGStab");


    // Constructors

        //- Construct from matrix components and solver controls
        metalPBiCGStab
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );


    //- Destructor
    virtual ~metalPBiCGStab() = default;


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
