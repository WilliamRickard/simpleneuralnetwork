# v24: safeguarded dense-tail quadratic backtracking

V24 keeps v23's optimiser, metric-balanced dense seed and 0.015% dense-BFGS
handover. The only retained optimisation change is how Armijo chooses its next
trial step after a rejection in the dense tail.

## Retained policy

For targets above 0.001%, v24 delegates directly to v23.

For targets at or below 0.001%, v24 follows v23 exactly through the L-BFGS/GN
stages and through the 0.015% dense handover. Once dense BFGS is active, each
line search still starts at `alpha = 1`.

If the first or a later dense-tail trial is rejected, v24 fits the one-point
quadratic model

\[
\alpha_q =
\frac{-g^T p\,\alpha^2}
{2\left[f(\theta+\alpha p)-f(\theta)-\alpha g^T p\right]}.
\]

A valid estimate is safeguarded to

\[
0.08\alpha \le \alpha_{next} \le 0.50\alpha.
\]

If the model is invalid or non-positive, v24 falls back to v23's half-step rule.
The Armijo acceptance test itself is unchanged.

## Why this helps

A rejected Armijo trial is expensive because it performs another complete
objective/gradient pass over the training rows. The dense matrix arithmetic is
small by comparison. V24 targets the rejected trials directly without changing
v23's pre-dense trajectory.

## Deterministic screening

Every objective/gradient evaluation and every Gauss-Newton calculation counts as
one full row-pass equivalent.

| Rows | Target | v23 passes | v24 passes | Reduction | v23 rejects | v24 rejects |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 0.001% | 578 | **505** | **12.6%** | 103 | **42** |
| 20,000 | 0.001% | 613 | **551** | **10.1%** | 84 | **30** |
| 100,000 | 0.001% | 558 | **500** | **10.4%** | 77 | **29** |
| 5,000 | 0.0005% | 1,045 | **996** | **4.7%** | 111 | **49** |
| 20,000 | 0.0005% | 1,268 | **1,183** | **6.7%** | 93 | **43** |
| 100,000 | 0.0005% | 987 | **926** | **6.2%** | 84 | **36** |

The unseen 10k and 50k row counts also improved at both targets without
retuning. The smallest reduction was 1.0% at 50k / 0.0005%.

## Reproducibility

`benchmark/dense_quadratic_benchmark.cpp` retains the known-good v23 research
harness structure because a cosmetically cleaned equivalent changed GCC's
floating-point trajectory in the deep tail. Mode `0` reproduces the merged v23
control and mode `8` enables the v24 dense-tail interpolation. The retained
safeguard is supplied as the final argument `0.08`.

Example:

```bash
g++ -std=c++11 -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -Wpedantic -Werror \
  v24/benchmark/dense_quadratic_benchmark.cpp -lmvec -o /tmp/v24-benchmark

/tmp/v24-benchmark 100000 0.0005 0 12000 5 0.08
/tmp/v24-benchmark 100000 0.0005 8 12000 5 0.08
```

The harness reduction order differs from the production AVX-512 evaluator, so
these are paired research comparisons rather than exact production iteration
claims.

## Validation

The exact v24 production wrapper compiles locally against v23 in both portable
and accelerated C++11 configurations with
`-O3 -Wall -Wextra -Wpedantic -Werror`.

The deterministic benchmark harness compiles with AVX2/FMA/OpenMP/libmvec under
the same strict warning policy and reproduces the published v23 controls before
showing the v24 reductions in `benchmark_summary.csv`.

`benchmark/experiments.txt` records rejected alternatives and the harness
reproducibility issue. Exploratory wall-clock measurements were noisy on the
shared host and are not used as the release gate.

GitHub Actions status must be checked separately. Local compilation is not a
claim that repository CI ran.
