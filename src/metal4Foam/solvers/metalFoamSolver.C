/*---------------------------------------------------------------------------*\
    metal4Foam: Metal-accelerated linear solvers for OpenFOAM (Apple GPU)

    Copyright (C) 2026 metal4Foam authors
    SPDX-License-Identifier: GPL-3.0-or-later

\*---------------------------------------------------------------------------*/

#include "metalFoamSolver.H"
#include "metalFoamBackend.H"
#include "PrecisionAdaptor.H"

#include <map>
#include <memory>
#include <sstream>

// The backend converts FP64 fields and int32 addressing directly:
static_assert
(
    sizeof(Foam::solveScalar) == 8,
    "metal4Foam requires a double-precision (WM_PRECISION_OPTION=DP) build"
);
static_assert
(
    sizeof(Foam::label) == 4,
    "metal4Foam requires a 32-bit label (WM_LABEL_SIZE=32) build"
);


// * * * * * * * * * * * * * * * * Constructor * * * * * * * * * * * * * * //

Foam::metalFoamSolver::metalFoamSolver
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    metalDict_(solverControls.subOrEmptyDict("metal"))
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::metalFoamBackend& Foam::metalFoamSolver::backend() const
{
    // One backend per (mesh, field): the lduMatrix::solver object itself
    // is re-created on every fvMatrix::solve, so cache here instead.
    static std::map<std::string, std::unique_ptr<metalFoamBackend>> cache;

    std::ostringstream key;
    key << fieldName_ << '@'
        << static_cast<const void*>(&matrix_.mesh());

    auto& slot = cache[key.str()];
    if (!slot)
    {
        slot.reset(new metalFoamBackend());
        Info<< "    " << slot->description() << nl;
    }
    return *slot;
}


Foam::solverPerformance Foam::metalFoamSolver::solveImpl
(
    const word& solverName,
    const bool useBiCGStab,
    solveScalarField& psi,
    const solveScalarField& source,
    const direction cmpt
) const
{
    // --- Coupled interfaces (processor/cyclic/AMI) are out of scope:
    //     their contributions cannot be applied inside the GPU-resident
    //     iteration without per-iteration CPU callbacks.
    forAll(interfaces_, patchi)
    {
        if (interfaces_.set(patchi))
        {
            FatalErrorInFunction
                << solverName << " does not support coupled boundaries "
                << "(processor/cyclic/AMI); run serially on a mesh "
                << "without coupled patches."
                << exit(FatalError);
        }
    }

    const label nCells = matrix_.mesh().lduAddr().size();

    solverPerformance solverPerf(solverName, fieldName_);

    solveScalarField pA(nCells);
    solveScalarField wA(nCells);

    // --- Initial FP64 residual: rA = source - A*psi (matches built-ins)
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    solveScalarField rA(source - wA);

    matrix().setResidualField
    (
        ConstPrecisionAdaptor<scalar, solveScalar>(rA)(),
        fieldName_,
        true
    );

    // --- FP64 normalisation factor (identical to built-in solvers)
    const solveScalar normFactor = this->normFactor(psi, source, wA, pA);

    if ((log_ >= 2) || (lduMatrix::debug >= 2))
    {
        Info<< "   Normalisation factor = " << normFactor << endl;
    }

    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm())/normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_, log_)
    )
    {
        metalFoamBackend& bk = backend();

        if (!bk.good())
        {
            FatalErrorInFunction
                << "Metal backend unavailable: " << bk.description()
                << exit(FatalError);
        }

        const lduAddressing& addr = matrix_.mesh().lduAddr();
        const label nFaces = addr.upperAddr().size();

        bk.setTopology
        (
            nCells,
            nFaces,
            addr.lowerAddr().cdata(),
            addr.upperAddr().cdata(),
            addr.ownerStartAddr().cdata(),
            addr.losortAddr().cdata(),
            addr.losortStartAddr().cdata()
        );

        // --- Row sums u = A*1 (FP64, incl. boundary contributions).
        //     A quasi-singular (Neumann + reference-cell) matrix has
        //     u ~ 0 in almost every row; FP32 CG on such a system keeps
        //     re-exciting the weakly-pinned constant mode and stalls.
        //     Cure: hand the GPU the exactly singular operator
        //     (diag0 = diag - u), solve in the zero-mean subspace, and
        //     correct the constant mode outside in FP64 (c-shift).
        solveScalarField u(nCells);
        matrix_.sumA(u, interfaceBouCoeffs_, interfaces_);

        const word deflateOpt =
            metalDict_.getOrDefault<word>("deflate", "auto");

        bool deflate = false;
        if (!useBiCGStab && deflateOpt != "off")
        {
            if (deflateOpt == "on")
            {
                deflate = true;
            }
            else
            {
                label nPinned = 0;
                const scalarField& dref = matrix_.diag();
                forAll(u, i)
                {
                    if (mag(u[i]) > 1e-8*mag(dref[i])) ++nPinned;
                }
                deflate = (10*nPinned <= nCells);
            }
        }

        scalarField diag0;
        const scalar* diagPtr = matrix_.diag().cdata();
        if (deflate)
        {
            diag0 = matrix_.diag();
            // fvm::laplacian produces negative diagonals: the deflated
            // diagonal must keep the original sign and stay away from 0
            forAll(diag0, i)
            {
                diag0[i] -= u[i];
                if (diag0[i]*matrix_.diag()[i] <= VSMALL*mag(diag0[i]))
                {
                    deflate = false;   // sign flip / zero: bail out
                    break;
                }
            }
            if (deflate) diagPtr = diag0.cdata();
        }

        const double tConvert = bk.setCoeffs
        (
            diagPtr,
            matrix_.upper().cdata(),
            (
                (!useBiCGStab || !matrix_.hasLower())
              ? nullptr                       // symmetric: lower == upper
              : matrix_.lower().cdata()
            )
        );

        // --- Chebyshev-Jacobi preconditioner setup (PCG only):
        //     Gershgorin upper bound for the spectrum of rD*A
        int chebOrder =
            useBiCGStab ? 1 : metalDict_.getOrDefault("chebyshevOrder", 8);
        const double chebRatio =
            metalDict_.getOrDefault<scalar>("chebyshevRatio", 30.0);
        double chebBeta = 2.0;
        if (chebOrder > 1)
        {
            const labelUList& lowA = addr.lowerAddr();
            const labelUList& uppA = addr.upperAddr();
            const scalarField& uppC = matrix_.upper();

            scalarField sumOff(nCells, Zero);
            forAll(uppA, f)
            {
                sumOff[lowA[f]] += mag(uppC[f]);
                sumOff[uppA[f]] += mag(uppC[f]);
            }
            // Spectrum bound of the Jacobi-scaled operator D^-1 A:
            // unit diagonal +- Gershgorin radius (sign-independent)
            chebBeta = 0;
            forAll(sumOff, i)
            {
                chebBeta =
                    Foam::max(chebBeta, 1.0 + sumOff[i]/mag(diagPtr[i]));
            }
        }

        // Absolute (unnormalised) L1 target for the FP32 inner solve
        const double targetResSum =
            Foam::max
            (
                solveScalar(tolerance_),
                solveScalar(relTol_)*solverPerf.initialResidual()
            )*normFactor;

        const int maxRestarts = metalDict_.getOrDefault("maxRestarts", 8);
        const int iterPerSync =
            metalDict_.getOrDefault("iterationsPerSync", 1);
        const bool profile = metalDict_.getOrDefault("profile", false);
        const bool verbose = metalDict_.getOrDefault("verbose", false);

        label totalIter = 0;
        double tGpu = 0;
        double tInner = 0;
        int outer = 0;
        bool singular = false;

        solveScalarField rin;   // projected inner RHS (deflated path)
        if (deflate) rin.resize(nCells);

        solveScalar prevRes = solverPerf.initialResidual();

        for (outer = 0; outer <= maxRestarts; ++outer)
        {
            // --- Inner RHS: zero-mean projection for the singular system
            const double* rPtr = rA.cdata();
            if (deflate)
            {
                solveScalar mean = 0;
                for (label i = 0; i < nCells; ++i) mean += rA[i];
                mean /= nCells;
                for (label i = 0; i < nCells; ++i) rin[i] = rA[i] - mean;
                rPtr = rin.cdata();
            }

            metalFoamBackend::Stats st;
            if (useBiCGStab)
            {
                st = bk.bicgstab
                (
                    rPtr,
                    targetResSum,
                    maxIter_ - totalIter
                );
            }
            else
            {
                st = bk.pcg
                (
                    rPtr,
                    targetResSum,
                    maxIter_ - totalIter,
                    iterPerSync,
                    chebOrder,
                    chebBeta,
                    chebRatio
                );
            }

            totalIter += st.nIterations;
            tGpu += st.tGpuSeconds;
            tInner += st.tWallSeconds;
            singular = st.singular;

            // --- FP64 update: psi += delta (zero-mean when deflated)
            const float* del = bk.delta();
            solveScalar delMean = 0;
            if (deflate)
            {
                for (label i = 0; i < nCells; ++i) delMean += del[i];
                delMean /= nCells;
            }
            solveScalar* psiPtr = psi.begin();
            for (label i = 0; i < nCells; ++i)
            {
                psiPtr[i] += solveScalar(del[i]) - delMean;
            }

            // --- Recompute true FP64 residual
            matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
            const solveScalar* srcPtr = source.cdata();
            const solveScalar* wAPtr = wA.cdata();
            solveScalar* rAPtr = rA.begin();
            for (label i = 0; i < nCells; ++i)
            {
                rAPtr[i] = srcPtr[i] - wAPtr[i];
            }

            // --- Constant-mode correction (FP64): psi += c with
            //     c = (u.r)/(u.u), exact residual update r -= c*u
            if (deflate)
            {
                solveScalar ur = 0, uu = 0;
                for (label i = 0; i < nCells; ++i)
                {
                    ur += u[i]*rAPtr[i];
                    uu += u[i]*u[i];
                }
                if (uu > VSMALL)
                {
                    const solveScalar c = ur/uu;
                    for (label i = 0; i < nCells; ++i)
                    {
                        psiPtr[i] += c;
                        rAPtr[i] -= c*u[i];
                    }
                }
            }

            solverPerf.finalResidual() =
                gSumMag(rA, matrix().mesh().comm())/normFactor;
            solverPerf.nIterations() = totalIter;

            // --- Hybrid polish: the deflated (exactly singular) operator
            //     converges fast but cannot zero the reference-cell defect
            //     it excludes. Once a deflated restart stops making
            //     progress, hand the GPU the exact operator to finish.
            if
            (
                deflate
             && solverPerf.finalResidual() > prevRes/1.5
             && !st.singular
            )
            {
                deflate = false;
                bk.setCoeffs
                (
                    matrix_.diag().cdata(),
                    matrix_.upper().cdata(),
                    nullptr
                );
                if (verbose)
                {
                    Info<< "    " << solverName
                        << " switching to exact operator (polish)" << nl;
                }
            }
            prevRes = solverPerf.finalResidual();

            if (verbose)
            {
                Info<< "    " << solverName << " restart " << outer
                    << ": inner=" << st.nIterations
                    << " fp32ResSum=" << st.finalResSum
                    << " fp64Res=" << solverPerf.finalResidual()
                    << (st.stagnated ? " (stagnated)" : "")
                    << (st.singular ? " (breakdown)" : "")
                    << nl;
            }

            if (singular) break;
            if (totalIter >= maxIter_) break;

            if
            (
                solverPerf.nIterations() >= minIter_
             && solverPerf.checkConvergence(tolerance_, relTol_, log_)
            )
            {
                break;
            }
        }

        if (profile)
        {
            Info<< "    " << solverName << " profile: cells=" << nCells
                << " iters=" << totalIter
                << " restarts=" << outer
                << " tConvert=" << 1000*tConvert << "ms"
                << " tInner=" << 1000*tInner << "ms"
                << " tGpu=" << 1000*tGpu << "ms"
                << nl;
        }
    }

    matrix().setResidualField
    (
        ConstPrecisionAdaptor<scalar, solveScalar>(rA)(),
        fieldName_,
        false
    );

    return solverPerf;
}

// ************************************************************************* //
