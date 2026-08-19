/*---------------------------------------------------------------------------*\
    metal4Foam: Metal-accelerated linear solvers for OpenFOAM (Apple GPU)

    Copyright (C) 2026 metal4Foam authors
    SPDX-License-Identifier: GPL-3.0-or-later

\*---------------------------------------------------------------------------*/

#include "metalPBiCGStab.H"
#include "PrecisionAdaptor.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(metalPBiCGStab, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<metalPBiCGStab>
        addmetalPBiCGStabSymMatrixConstructorToTable_;

    lduMatrix::solver::addasymMatrixConstructorToTable<metalPBiCGStab>
        addmetalPBiCGStabAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * //

Foam::metalPBiCGStab::metalPBiCGStab
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

Foam::solverPerformance Foam::metalPBiCGStab::scalarSolve
(
    solveScalarField& psi,
    const solveScalarField& source,
    const direction cmpt
) const
{
    return solveImpl(typeName, true, psi, source, cmpt);
}


Foam::solverPerformance Foam::metalPBiCGStab::solve
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
