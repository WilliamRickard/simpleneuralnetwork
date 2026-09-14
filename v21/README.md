# v21: target-aware Gauss-Newton scale refresh

V21 builds on v20's deep-stage Gauss-Newton diagonal preconditioner. V20 computes the diagonal once when percentage error first reaches 0.05% and keeps it fixed. V21 keeps that behaviour for every requested target above 0.001%, then refreshes the same positive diagonal at a small number of later error levels only when an ultra-deep target is actually requested.

## Retained change

For targets above 0.001%, `trainRangeV21` delegates directly to `trainRangeV20`. The v20 algorithm and trajectory are therefore retained rather than reimplemented.

For a target at or below 0.001%, v21 uses the normal v20 scale at 0.05% PE and recomputes it when the current PE first reaches:

- 0.0225%;
- 0.01%.

For a target at or below 0.0005%, one additional refresh is performed at 0.002% PE.

Each refresh recomputes exactly the same v20 scale

\[
s_i=\operatorname{clip}\left(\sqrt{\frac{m}{\max(d_i,10^{-18}m)}},\frac{1}{28},28\right),
\]

where \(d_i\) is the mean squared prediction Jacobian for parameter \(i\) and \(m\) is the median of the 192 diagonal entries. No running average or new curvature formula is introduced.

## Why refresh the diagonal

The v20 diagonal is local curvature information. It materially improves the deep L-BFGS phase, but the weights continue to move after the 0.05% state at which the scale was measured. Screening showed that recomputing the same diagonal at a few later points can substantially shorten the tail.

The refreshes are intentionally sparse. Recomputing too late or too frequently was trajectory-sensitive, and extra full-data passes can erase the benefit. The retained thresholds were selected for cross-scale robustness rather than as the single fastest setting on one row count.

The general approach is consistent with published L-BFGS work showing that diagonal preconditioning and diagonal BFGS-style initial matrices can materially improve ill-conditioned limited-memory optimisation. V21 does not implement those papers' update formulae; it retains v20's measured Gauss-Newton diagonal and only refreshes it.

## Structural preservation

V21 retains all of the following from v20:

- FP64 production weights, activations, gradients and deltas;
- the v17 objective/gradient evaluator;
- v18 scale-aware curvature acceptance;
- v19 history 10 to history 160 transition at 0.05%;
- v20 median-normalised square-root inverse Gauss-Newton scale and clip 28;
- Armijo backtracking and its constants;
- the MSE training objective;
- percentage-error stopping semantics;
- scalar FP64 final confirmation.

For targets above 0.001%, v21 calls `trainRangeV20` directly. There is no v21 refresh code on those runs.

## Strict deterministic screening

The retained policy was tested with `benchmark/refresh_benchmark.cpp`, a four/five-worker FP64 AVX2/libmvec implementation of the deterministic 11-16-1 benchmark. Results below were compiled with `-O3` and without `-ffast-math`. The screening reduction order differs from the production AVX-512 evaluator, so exact production iteration counts are not claimed.

Every objective/gradient evaluation is counted as one full row pass. Every Gauss-Newton diagonal computation is also counted as one full row pass.

| Rows | Target | v20 passes | v21 passes | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 5,000 | 0.001% | 2,781 | 1,211 | 56.5% |
| 20,000 | 0.001% | 1,829 | 1,166 | 36.2% |
| 100,000 | 0.001% | 1,227 | 998 | 18.7% |
| 5,000 | 0.0005% | 6,228 | 2,248 | 63.9% |
| 20,000 | 0.0005% | 7,431 | 3,812 | 48.7% |
| 100,000 | 0.0005% | 3,526 | 2,593 | 26.5% |

The gain is therefore smaller at 100k than at 5k/20k but remains positive at both retained targets.

## Rejected refresh policies

- Refreshing at 0.005% as the first extra refresh was too late and could worsen the tail.
- A single refresh around 0.02-0.03% helped some scales but was not as robust as the retained two-stage schedule.
- Moving the second refresh away from 0.01% to 0.011%, 0.009% or 0.0075% materially worsened the 5k 0.001% screen.
- Enabling the extra refreshes for a 0.002% requested target could regress the 5k case. This is why v21 delegates directly to v20 whenever the requested target is above 0.001%.
- For 0.0005% targets, a third refresh at 0.002% was consistently beneficial across 5k, 20k and 100k. Refreshes at 0.005% or 0.003% were weaker.

## Validation

`v21/main.cpp` was compiled locally against the inherited v20 interface in both portable and accelerated C++11 configurations with `-O3 -Wall -Wextra -Wpedantic -Werror`. The benchmark harness was also compiled under the strict warning policy.

GitHub Actions status must be checked separately. A local compile is not a claim that repository CI ran.
