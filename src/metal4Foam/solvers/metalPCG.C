/*---------------------------------------------------------------------------*\
    metal4Foam: Metal-accelerated linear solvers for OpenFOAM (Apple GPU)

    Copyright (C) 2026 metal4Foam authors
    SPDX-License-Identifier: GPL-3.0-or-later

\*---------------------------------------------------------------------------*/

#include "metalPCG.H"
#include "PrecisionAdaptor.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(metalPCG, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<metalPCG>
        addmetalPCGSymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * //

Foam::metalPCG::metalPCG
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    metalFoamSolver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * //

Foam::solverPerformance Foam::metalPCG::scalarSolve
(
    solveScalarField& psi,
    const solveScalarField& source,
    const direction cmpt
) const
{
    return solveImpl(typeName, false, psi, source, cmpt);
}


Foam::solverPerformance Foam::metalPCG::solve
(
    scalarField& psi_s,
    const scalarField& source,
    const direction cmpt
) const
{
    PrecisionAdaptor<solveScalar, scalar> tpsi(psi_s);
    return scalarSolve
    (
        tpsi.ref(),
        ConstPrecisionAdaptor<solveScalar, scalar>(source)(),
        cmpt
    );
}

// ************************************************************************* //
