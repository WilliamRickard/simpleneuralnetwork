# v22: dense BFGS final-tail solver

V22 keeps v21 unchanged for ordinary targets and replaces limited-memory BFGS
with full-memory BFGS only in the ultra-deep final tail.

## Retained policy

For requested percentage-error targets above **0.001%**, `trainRangeV22`
delegates directly to `trainRangeV21`.

For targets at or below **0.001%**, v22 follows v21 normally until the current
percentage error reaches **0.015%**. At that point it constructs a dense
192 x 192 inverse-Hessian approximation and uses full BFGS for the remaining
optimisation.

The dense matrix is initialised as

\[
H_0 = \gamma\,\operatorname{diag}(q_1,\ldots,q_{192}),
\]

where `q` is v21's most recent Gauss-Newton diagonal scale and

\[
\gamma=\frac{s^T y}{y^T y}
\]

is the ordinary L-BFGS scalar initial-Hessian multiplier from the newest
accepted secant pair.

Each accepted dense-stage pair then applies the standard inverse-BFGS update

\[
H^+ = H
+ \left(1+\frac{y^T H y}{s^T y}\right)\frac{ss^T}{s^T y}
- \frac{Hys^T+sy^T H}{s^T y}.
\]

The existing v18 scale-aware curvature safeguard remains mandatory before this
update is applied.

## Why full BFGS is practical here

The network has only 192 parameters. A dense FP64 192 x 192 matrix contains
36,864 doubles, about 288 KiB. Its matrix-vector product and rank-two update are
small compared with another full pass over 5,000 to 100,000 training rows.

V21 already permits 160 L-BFGS curvature pairs in the deep stage. Earlier
profiling showed that this history becomes highly redundant and ill-conditioned.
At this parameter count, retaining the complete BFGS matrix is inexpensive and
avoids limited-memory truncation precisely where convergence is slowest.

## Structural preservation

V22 retains from v21:

- FP64 production weights, activations, gradients and deltas;
- v17 objective/gradient evaluation and final scalar confirmation;
- v18 scale-aware curvature acceptance;
- v19 history staging before the dense transition;
- v20 median-normalised square-root inverse Gauss-Newton preconditioning;
- v21 target-aware GN refreshes before the dense transition;
- Armijo backtracking, MSE objective and percentage-error stopping semantics.

After dense BFGS takes over, later v21 GN refreshes are deliberately skipped
because their diagonal scale is no longer used.

## Strict deterministic screening

The benchmark harness uses the deterministic v21 11-16-1 problem and five
worker slices, which reproduce the merged v21 benchmark counts exactly. It was
compiled with `-O3` and without `-ffast-math`. Every objective/gradient
evaluation and every Gauss-Newton calculation counts as one row-pass equivalent.

| Rows | Target | v21 passes | v22 passes | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 5,000 | 0.001% | 1,211 | **793** | **34.5%** |
| 20,000 | 0.001% | 1,166 | **857** | **26.5%** |
| 100,000 | 0.001% | 998 | **822** | **17.6%** |
| 5,000 | 0.0005% | 2,248 | **1,367** | **39.2%** |
| 20,000 | 0.0005% | 3,812 | **1,607** | **57.8%** |
| 100,000 | 0.0005% | 2,593 | **1,804** | **30.4%** |

The harness reduction order differs from the production AVX-512 evaluator, so
these are paired research comparisons rather than exact production iteration
claims.

## Alternatives rejected for v22

The same harness screened dynamic diagonal BFGS, block Gauss-Newton
preconditioning and a strong-Wolfe line search. The dynamic diagonal method
stalled, while block-GN and strong Wolfe were not robust across row counts.
`benchmark/experiments.txt` records those results and the dense-switch study.

## Validation

The exact production source was compiled locally against the inherited v21
interface in portable and accelerated C++11 configurations with
`-O3 -Wall -Wextra -Wpedantic -Werror`. The benchmark harness was also compiled
under the strict warning policy.

GitHub Actions status must be checked independently; local compilation is not a
claim that repository CI ran.
