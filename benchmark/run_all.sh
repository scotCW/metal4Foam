#!/bin/bash
# Full benchmark sweep: cavity at several mesh sizes, 20 time steps,
# pressure solver = CPU PCG/DIC, CPU PCG/diagonal, metalPCG.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
STEPS=${STEPS:-20}
SIZES=${SIZES:-"100 200 400 800 1200"}
SOLVERS=${SOLVERS:-"pcg-dic pcg-diag metal"}
OUT="$HERE/results_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "# metal4Foam cavity benchmark"
    echo "# host: $(sysctl -n machdep.cpu.brand_string), \
$(sysctl -n hw.memsize | awk '{print $1/1073741824}') GB unified memory"
    echo "# $(sw_vers -productName) $(sw_vers -productVersion), \
OpenFOAM-v2506 darwin64ClangDPInt32Opt (arm64 native)"
    echo "# steps=$STEPS  (dt scaled to keep Co=1)"
} | tee "$OUT"

for N in $SIZES; do
    for S in $SOLVERS; do
        "$HERE/run_case.sh" "$N" "$S" "$STEPS" "bench_${N}_${S}" \
            2>&1 | grep -v '^real\|^user\|^sys' | tee -a "$OUT"
    done
done
echo "results in $OUT"
