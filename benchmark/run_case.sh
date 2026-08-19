#!/bin/bash
# run_case.sh <N> <solver: pcg-dic|pcg-diag|metal|pbicgstab|metal-bicg> <steps> [tag]
# Builds an NxN cavity case, runs icoFoam serially, leaves the run in
# runs/<tag-or-auto>/ with log.icoFoam.
set -e
N=${1:?N}
SOLVER=${2:?solver}
STEPS=${3:?steps}
TAG=${4:-cavity_${N}_${SOLVER}}

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$WM_PROJECT_DIR/tutorials/incompressible/icoFoam/cavity/cavity"
RUN="$HERE/runs/$TAG"

rm -rf "$RUN"
mkdir -p "$HERE/runs"
cp -R "$SRC" "$RUN"
cd "$RUN"

# --- mesh resolution
sed -i.bak "s/(20 20 1)/($N $N 1)/" system/blockMeshDict

# --- time step: keep Courant number at the tutorial's Co=1
DT=$(python3 -c "print(0.005*20/$N)")
ET=$(python3 -c "print($STEPS*0.005*20/$N)")
# --- viscosity: keep the diffusion number nu*dt/dx^2 at the tutorial's
#     value too (Re = 0.5*N, laminar for the lid-driven cavity)
NU=$(python3 -c "print(0.01*20/$N)")
sed -i.bak "s/0\.01;/$NU;/" constant/transportProperties
WI=$STEPS   # write only the final step

sed -i.bak \
    -e "s/^deltaT.*/deltaT          $DT;/" \
    -e "s/^endTime.*/endTime         $ET;/" \
    -e "s/^writeControl.*/writeControl    timeStep;/" \
    -e "s/^writeInterval.*/writeInterval   $WI;/" \
    -e "s/^writePrecision.*/writePrecision  12;/" \
    system/controlDict

# --- plugin library + profiling instrumentation
cat >> system/controlDict <<'EOF'

libs            (metalFoam);

profiling
{
    active      true;
    cpuInfo     false;
    memInfo     false;
    sysInfo     false;
}
EOF

# --- pressure solver selection
case "$SOLVER" in
pcg-dic)
    : ;;  # tutorial default: PCG + DIC
pcg-diag)
    sed -i.bak "s/preconditioner  DIC;/preconditioner  diagonal;/" \
        system/fvSolution ;;
pbicgstab)
    sed -i.bak -e "s/solver          PCG;/solver          PBiCGStab;/" \
        -e "s/preconditioner  DIC;/preconditioner  diagonal;/" \
        system/fvSolution ;;
metal)
    sed -i.bak -e "s/solver          PCG;/solver          metalPCG;/" \
        -e "/preconditioner  DIC;/d" system/fvSolution
    sed -i.bak "/relTol          0.05;/a\\
        metal { profile true; }" system/fvSolution ;;
metal-bicg)
    sed -i.bak -e "s/solver          PCG;/solver          metalPBiCGStab;/" \
        -e "/preconditioner  DIC;/d" system/fvSolution
    sed -i.bak "/relTol          0.05;/a\\
        metal { profile true; }" system/fvSolution ;;
u-bicg)
    # momentum (asymmetric) on the GPU; pressure stays on CPU PCG/DIC
    sed -i.bak -e "s/solver          smoothSolver;/solver          metalPBiCGStab;/" \
        -e "/smoother        symGaussSeidel;/d" system/fvSolution ;;
u-pbicgstab)
    # CPU reference for the asymmetric comparison
    sed -i.bak -e "s/solver          smoothSolver;/solver          PBiCGStab;/" \
        -e "s/smoother        symGaussSeidel;/preconditioner  diagonal;/" system/fvSolution ;;
*)
    echo "unknown solver $SOLVER"; exit 1 ;;
esac

blockMesh > log.blockMesh 2>&1
/usr/bin/time -p icoFoam > log.icoFoam 2>&1 || { tail -20 log.icoFoam; exit 1; }

# --- quick summary
CLOCK=$(grep -o 'ClockTime = [0-9.]*' log.icoFoam | tail -1)
EXEC=$(grep -o 'ExecutionTime = [0-9.]*' log.icoFoam | tail -1)
PITER=$(grep 'Solving for p' log.icoFoam | grep -o 'No Iterations [0-9]*' \
        | awk '{s+=$3} END {print s}')
PROF=$(find . -name profiling -path '*/uniform/*' | tail -1)
PSOLVE=$(python3 - "$PROF" <<'PYEOF'
import re, sys
txt = open(sys.argv[1]).read()
for m in re.finditer(r'description\s+"fvMatrix::solve\.(\w+)";\s*'
                     r'calls\s+(\d+);\s*totalTime\s+([0-9.e+-]+);', txt):
    f, c, t = m.group(1), int(m.group(2)), float(m.group(3))
    print(f"solve.{f}: calls={c} total={t:.4f}s per-call={1000*t/c:.3f}ms")
PYEOF
)
echo "$TAG: cells=$((N*N)) steps=$STEPS $EXEC $CLOCK totalPIters=$PITER"
echo "$PSOLVE"
