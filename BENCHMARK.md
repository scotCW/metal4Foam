# metal4Foam — validation and benchmark report

**Hardware:** Apple M3 Pro (11-core CPU, 14-core GPU), 36 GB unified memory
**OS / toolchain:** macOS 15.7.7, Xcode Metal toolchain, OpenFOAM-v2506
`darwin64ClangDPInt32Opt` — native arm64, verified with `file` (no Rosetta):

```
platforms/darwin64ClangDPInt32Opt/bin/icoFoam:      Mach-O 64-bit executable arm64
platforms/darwin64ClangDPInt32Opt/lib/libmetalFoam.dylib: Mach-O 64-bit dynamically linked shared library arm64
```

**Case:** `incompressible/icoFoam/cavity`, serial, single precision solve on
GPU with FP64 refinement. Mesh refined from 20×20 to 1200×1200; `deltaT`
scaled to hold Courant number at the tutorial value and `nu` scaled to hold
the diffusion number (so `Re = 0.5·N` — every solver sees a stable case).
Timings are OpenFOAM's own `profiling` output for `fvMatrix::solve.p`,
which isolates the linear solve from discretisation and I/O.

---

## 1. Correctness validation

### Phase 1 — `metalPCG` vs built-in `PCG` (cavity 20×20, 100 steps, t=0.5)

| Comparison | p relMax | U<sub>x</sub> relMax | U<sub>y</sub> relMax |
|---|---|---|---|
| **metalPCG vs PCG/DIC** (tutorial default) | 1.78e-06 | 3.50e-07 | 8.03e-07 |
| **metalPCG vs PCG/diagonal** | 1.78e-06 | 3.50e-07 | 8.03e-07 |
| *CPU-vs-CPU floor* (DIC vs diagonal) | *4.89e-08* | *2.10e-08* | *6.85e-08* |

The GPU solver agrees with both CPU references to ~2e-6 relative — the
per-solve convergence tolerance (`tolerance 1e-6`, `relTol 0.05`), i.e. the
difference is dominated by where each solver stops, not by FP32 arithmetic.
The CPU-vs-CPU row is the honest noise floor: two *CPU* solvers differing
only in preconditioner disagree by 5e-8, so metal4Foam sits ~35x above that
floor but ~3 orders below anything physically meaningful.

Residual traces track the CPU solver closely — e.g. first `pFinal` solve:
CPU diagonal-PCG `Initial 0.528654 → Final 8.68e-07` in 101 iterations,
metalPCG `Initial 0.528654 → Final 6.69e-07` in 104 iterations.

### Phase 2 — `metalPBiCGStab` vs built-in `PBiCGStab` (momentum, 200×200)

Identical iteration counts, matching residuals:

| Solve | CPU `diagonalPBiCGStab` | `metalPBiCGStab` |
|---|---|---|
| U<sub>x</sub> #1 | init 1, final 7.02e-06, **17 iters** | init 1, final 7.24e-06, **17 iters** |
| U<sub>x</sub> #2 | init 0.187823091, final 9.596e-06, **15 iters** | init 0.187823118, final 9.594e-06, **15 iters** |
| U<sub>x</sub> #3 | init 0.081646723, final 6.164e-06, **14 iters** | init 0.081646735, final 6.174e-06, **14 iters** |

Final fields (t=0.01): p relMax **6.87e-08**, U<sub>x</sub> relMax
**6.82e-08**, U<sub>y</sub> relMax **9.42e-08** — at the CPU-vs-CPU floor.

---

## 2. Benchmark: pressure solve, per call

`fvMatrix::solve.p` total / calls, 20 time steps (40 pressure solves).

| Mesh | Cells | CPU `PCG`+DIC | CPU `PCG`+diag | **`metalPCG`** | speedup vs DIC |
|---|---:|---:|---:|---:|---:|
| 20² | 400 | 0.066 ms | — | 1.109 ms | **0.06x** (16.8x slower) |
| 50² | 2,500 | 2.082 ms | — | 5.838 ms | **0.36x** (2.8x slower) |
| 70² | 4,900 | 6.434 ms | — | 8.331 ms | **0.77x** (1.3x slower) |
| 100² | 10,000 | 20.52 ms | 32.19 ms | **11.63 ms** | **1.76x** |
| 200² | 40,000 | 173.8 ms | 267.7 ms | **34.40 ms** | **5.05x** |
| 400² | 160,000 | 1500.8 ms | *diverged* | **249.7 ms** | **6.01x** |
| 800² | 640,000 | 10508.3 ms | *4982.3 ms (unconverged)* | **1787.8 ms** | **5.88x** |
| 1200² | 1,440,000 | *did not converge* | *did not converge* | *did not converge* | — |

### The crossover is ~7,000 cells

Below ~5k cells the GPU is **slower**, and at 400 cells it is slower by
almost 17x. This is exactly the expected result and worth stating plainly:
each Krylov iteration ends in a `waitUntilCompleted` so the CPU can test
convergence, and that round-trip costs ~0.1–0.3 ms regardless of mesh size.
At 400 cells a CPU PCG iteration costs microseconds, so the sync dominates
completely. The GPU only wins once there is enough arithmetic per iteration
to hide the sync — measured between 4,900 cells (still 1.3x slower) and
10,000 cells (1.76x faster).

Speedup saturates around **6x** from 160k cells upward, which is consistent
with this being a **bandwidth-bound** problem on a **shared memory bus**:
the M3 Pro's GPU has more bandwidth available to it than a single CPU core,
but not unlimited, and both processors are drawing on the same 36 GB pool.
This is the flip side of the unified-memory advantage — no transfer cost,
but no separate memory bandwidth either.

### Iteration counts (total over 20 steps) — read with care

| Mesh | DIC | diagonal | metalPCG |
|---|---:|---:|---:|
| 100² | 4,915 | 16,843 | 1,851 |
| 200² | 10,298 | 34,131 | 4,240 |
| 400² | 21,585 | *maxIter* | 11,599 |
| 800² | 37,310 | *40,000 (maxIter)* | 20,230 |

metalPCG's iteration count looks better than DIC's, but **one metalPCG
iteration is not one DIC iteration**: with `chebyshevOrder 8` each
iteration performs 8 SpMVs. In SpMV-equivalent terms metalPCG at 100² does
~14.8k SpMVs vs DIC's ~4.9k — it does roughly **3x more arithmetic** and
still wins on wall clock, because that arithmetic is parallel and needs no
extra CPU syncs. The relevant comparison against DIC is wall time, not
iterations.

The like-for-like *preconditioner* comparison is the diagonal column: same
Jacobi-class preconditioning on both sides, where metalPCG is **2.8x
faster at 100² and 7.8x at 200²** — and converges at 400²/800² where CPU
diagonal-PCG does not.

## 3. Momentum (asymmetric) benchmark — GPU loses here

Cavity 200² (40k cells), `fvMatrix::solve.U`:

| Solver | per call |
|---|---:|
| CPU `PBiCGStab` + diagonal | **11.16 ms** |
| `metalPBiCGStab` | 13.75 ms (**1.23x slower**) |

Honest result: **the momentum equation is not worth offloading.** The
transient term makes those matrices strongly diagonally dominant, so they
converge in ~15 iterations — and BiCGStab needs *two* CPU syncs per
iteration (to reproduce the built-in solver's mid-iteration exit test on
`|sA|`), so ~30 syncs of overhead sit on top of very little arithmetic.
Phase 2 is correct and validated, but its value is asymmetric *scalar
transport* on large meshes, not momentum on typical ones.

---

## 4. Where it fails, and why

### FP32 precision floor at ~1.4M cells

At 1200² the solver does not converge, and the diagnosis is unambiguous
from the verbose trace:

```
metalPCG restart 0: inner=51 fp32ResSum=2.83e-06 fp64Res=1.272 (stagnated)
metalPCG restart 1: inner=51 fp32ResSum=3.24e-06 fp64Res=1.460 (stagnated)
...
```

The GPU reports its own residual as ~3e-6 while the true FP64 residual is
~1.2 — the FP32 recurrence has completely decoupled from reality. Cause:
at 1.44M cells the converged per-cell residual is ~1e-13 while the
individual SpMV terms being summed are ~1e-6. Recovering a result 7 orders
below the magnitude of its own operands consumes FP32's entire ~7-digit
budget, so the SpMV returns noise. Outer FP64 refinement cannot rescue this
because *every* restart hits the same wall on the fresh residual.

This is a floor of the FP32-inner-solve architecture, not a tunable bug.
Metal has no FP64 in shaders, so the fixes would be FP32 with compensated
(Kahan/two-sum) accumulation in the SpMV, or a mixed-precision formulation
that keeps the residual well-scaled — both substantial work, and neither
attempted here.

Note this is **not** a case where the CPU quietly wins: at 1200² CPU
PCG/DIC also fails to converge (`Final residual 0.2548, 1000 iterations`),
and CPU diagonal-PCG diverges outright (`Final residual 7.19`). At this
mesh size the honest answer for all three solvers is *use GAMG* — which is
deliberately out of scope.

### Partial under-convergence at 640k cells

At 800² some intermediate `p` solves stop above their `relTol` target
(e.g. `Initial 0.0632 → Final 0.0128` in 503 iterations) — early symptoms
of the same FP32 floor. The `pFinal` solves still reach tolerance
(3.36e-06, 5.57e-07) and the time-step continuity error is **2.9e-11**, so
the solution remains sound; but 640k cells is clearly near the edge.

### Other limitations

- **Serial only.** Coupled interfaces (processor/cyclic/AMI) raise a fatal
  error with a clear message rather than silently producing wrong answers.
- **No GAMG equivalent** — out of scope by instruction. For meshes where
  only multigrid converges, GAMG on CPU remains the right tool. This
  benchmark shows that boundary is real and arrives around 1M cells.
- **Jacobi-class preconditioning only.** DIC/DILU are inherently sequential
  triangular sweeps; the Chebyshev polynomial preconditioner is the GPU-
  friendly substitute and closes most (not all) of the gap.

---

## 5. Reproducing

```bash
source /Volumes/OpenFOAM/of-env.sh
cd metal4Foam && ./Allwmake
cd benchmark && ./run_all.sh                      # full sweep
./run_case.sh 200 metal 20 mycase                 # single case
./compare_fields.py runs/A runs/B 0.5 p U         # field comparison
```
