# v23: metric-balanced dense BFGS scaling

V23 keeps v22's optimiser structure and 0.015% dense-BFGS handover. The only
production optimisation change is how the initial dense inverse-Hessian matrix
is scaled at that handover.

## Retained policy

For requested percentage-error targets above **0.001%**, `trainRangeV23`
delegates directly to v22.

For targets at or below **0.001%**, v23 follows v22 exactly until PE reaches
**0.015%**. It then initialises the same full 192 x 192 inverse-BFGS matrix from
v21's current Gauss-Newton diagonal scale `D`, but replaces v22's identity-based
scalar with a scale-aware one.

V22 uses

\[
H_0 = \gamma D,
\qquad
\gamma_{22}=\frac{s^T y}{y^T y}.
\]

The scalar \(\gamma_{22}\) is the standard L-BFGS scaling for a multiple of the
identity. Once the seed is a nontrivial diagonal matrix, a more symmetric choice
is available.

V23 chooses \(\gamma\) so that the newest accepted secant has equal quadratic
size in the primal and dual metrics:

\[
s^T H_0^{-1}s = y^T H_0 y.
\]

Substituting \(H_0=\gamma D\) gives

\[
\boxed{
\gamma_{23}=
\sqrt{\frac{s^T D^{-1}s}{y^T D y}}
}.
\]

The resulting seed remains

\[
H_0=\gamma_{23}D.
\]

No additional dataset pass is required. The calculation uses the newest accepted
secant pair and the GN diagonal already held by v22. If the scale cannot be
computed safely, production falls back to v22's scalar.

## What remains unchanged

V23 retains:

- v22's **0.015%** dense-BFGS transition;
- direct delegation to v22 for requested targets above **0.001%**;
- the same 192 x 192 dense inverse-BFGS update;
- v18 scale-aware curvature acceptance;
- v21/v20 Gauss-Newton diagonal construction before the dense handover;
- v19 staged L-BFGS history before the handover;
- Armijo backtracking;
- the MSE objective and percentage-error stopping rule;
- FP64 production arithmetic and scalar final confirmation.

The adaptive-switch research initially planned for v23 was not retained. Multiple
history-rank, redundancy, conditioning and held-out secant metrics were less
robust across row counts than v22's existing transition. Keeping the reliable
switch and fixing the dense seed was both simpler and faster.

## Strict deterministic screening

The retained five-worker AVX2/libmvec harness reproduces merged v22 exactly.
Every objective/gradient evaluation and every Gauss-Newton calculation counts as
one full row-pass equivalent.

| Rows | Target | v22 passes | v23 passes | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 5,000 | 0.001% | 793 | **578** | **27.1%** |
| 20,000 | 0.001% | 857 | **613** | **28.5%** |
| 100,000 | 0.001% | 822 | **558** | **32.1%** |
| 5,000 | 0.0005% | 1,367 | **1,045** | **23.6%** |
| 20,000 | 0.0005% | 1,607 | **1,268** | **21.1%** |
| 100,000 | 0.0005% | 1,804 | **987** | **45.3%** |

Unseen 10k and 50k row counts were also tested without retuning. All four
row-count/target combinations improved; the smallest gain was 7.6% at
50k / 0.0005%.

The harness reduction order differs from the production AVX-512 evaluator, so
these are paired research comparisons rather than exact production iteration
claims.

## Paired wall-clock benchmark

A separate timing benchmark ran 10 measured repetitions of v21, v22 and v23 for
each retained row-count/target case, after one discarded warm-up. Variant order
rotated between repetitions so the three versions were measured adjacent in
time rather than in long version-specific blocks.

The benchmark host exposed five Intel Xeon Platinum 8573C cores. The harness was
compiled with `g++ 14.2.0`, C++11, `-O3 -mavx2 -mfma -fopenmp` and libmvec,
without `-ffast-math`. Timings use `std::chrono::steady_clock` and the same five
deterministic worker slices as the release screen.

Median optimiser times, with p10-p90 in brackets:

| Rows | Target | v21 | v22 | v23 | v23 vs v22 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 0.001% | 0.358 s [0.336, 0.429] | 0.234 s [0.195, 0.249] | **0.140 s [0.121, 0.149]** | **1.71x** |
| 20,000 | 0.001% | 1.020 s [0.976, 1.190] | 0.763 s [0.710, 1.034] | **0.512 s [0.468, 0.566]** | **1.48x** |
| 100,000 | 0.001% | 3.717 s [3.540, 4.047] | 2.908 s [2.874, 3.472] | **1.991 s [1.935, 2.135]** | **1.49x** |
| 5,000 | 0.0005% | 0.718 s [0.679, 0.827] | 0.392 s [0.376, 0.636] | **0.283 s [0.263, 0.343]** | **1.42x** |
| 20,000 | 0.0005% | 3.289 s [3.197, 3.724] | 1.341 s [1.256, 2.245] | **1.045 s [1.005, 1.066]** | **1.30x** |
| 100,000 | 0.0005% | 9.453 s [9.207, 11.423] | 6.822 s [6.367, 7.252] | **3.548 s [3.413, 4.088]** | **1.88x** |

V23's paired median wall-clock speedup over v22 ranges from **1.30x to 1.88x**
and over v21 from **1.85x to 3.12x**. Even the p10 paired v23-v22 speedup is
above 1.18x in every retained case.

The host is shared rather than CPU-isolated, so wall-clock data are inherently
noisier than deterministic row-pass counts. No slow observations were removed.
See `benchmark/timing_methodology.md`, `timing_raw.csv`, `timing_summary.csv` and
`timing_speedups.csv` for the full methodology and results.

## Validation

The exact production source was compiled locally against the inherited v22
interface in portable and accelerated C++11 configurations with
`-O3 -Wall -Wextra -Wpedantic -Werror`.

`benchmark/scale_aware_gamma_benchmark.cpp` also compiles under the strict
warning policy with AVX2, FMA, OpenMP and libmvec, and reproduces all retained
v22/v23 counts in `benchmark_summary.csv`.

`benchmark/timing_benchmark.cpp` compiles under the same strict AVX2/FMA/OpenMP/
libmvec warning policy and reproduces the deterministic pass counts while adding
only `std::chrono::steady_clock` instrumentation.

GitHub Actions status must be checked separately. Local compilation is not a
claim that repository CI ran.
