# v20: deep-stage Gauss-Newton diagonal preconditioning

V20 keeps the v19 optimiser exactly through 0.05% percentage error. When a deeper target requires another iteration, it computes one Gauss-Newton diagonal pass and uses that fixed positive diagonal only as the initial inverse-Hessian scale inside the existing 160-pair L-BFGS two-loop recursion.

## Retained change

For parameter \(\theta_i\), v20 computes the diagonal Gauss-Newton quantity

\[
d_i = \frac{1}{n}\sum_r \left(\frac{\partial \hat y_r}{\partial \theta_i}\right)^2.
\]

Let \(m\) be the median of the 192 diagonal entries. The scale used in the deep stage is

\[
s_i = \operatorname{clip}\left(\sqrt{\frac{m}{\max(d_i,10^{-18}m)}},\frac{1}{28},28\right).
\]

V19's scalar L-BFGS initial scale \(\gamma\) is therefore replaced by the positive diagonal \(\gamma s_i\) only after the deep stage begins. The square root deliberately tempers the raw inverse-diagonal conditioning correction.

The diagonal is computed once, at the first state already at or below 0.05% PE. It is then held fixed. A target of 0.05% or above exits before this pass is performed, so the established v19 fast path is structurally unchanged.

## What remains unchanged

- FP64 weights, activations, gradients and deltas
- v17's production objective/gradient evaluator
- v18's scale-aware positive-curvature acceptance
- v19's history 10 to history 160 deep-stage transition
- Armijo backtracking and its constants
- MSE training objective
- percentage-error stopping semantics
- scalar FP64 final confirmation

## Why diagonal preconditioning

V19 diagnostics showed that the 160-pair history becomes highly redundant in the deep tail, but v20 screening found that direct pair rejection, pair replacement, age-aware eviction, two-timescale memory and a late 192-pair expansion were trajectory-sensitive and did not improve consistently across row counts.

A Gauss-Newton diagnostic at low error instead showed severe coordinate conditioning. Pair-selection heuristics therefore target a symptom, while a diagonal initial inverse-Hessian scale addresses the measured anisotropy without discarding secant information or changing chronological history.

## Deterministic C++ screening

The retained policy was screened with a separate FP64 four-worker AVX2/libmvec implementation of the deterministic 11-16-1 benchmark. Its reduction order differs from the production AVX-512 evaluator, so exact counts are not expected to match production bit-for-bit. The comparison is within the same harness and counts the one Gauss-Newton pass explicitly as one additional row-pass equivalent.

| Rows | Target | v19 FIFO passes | v20 passes | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 5,000 | 0.005% | 858 | 393 | 54.2% |
| 5,000 | 0.001% | 4,423 | 1,672 | 62.2% |
| 20,000 | 0.005% | 612 | 339 | 44.6% |
| 20,000 | 0.001% | 3,700 | 1,796 | 51.5% |
| 100,000 | 0.005% | 591 | 333 | 43.7% |
| 100,000 | 0.001% | 3,643 | 1,199 | 67.1% |

The retained clip of 28 is a robustness choice rather than the single best setting at every row count. Larger clips could be faster on one screen and worse on another. Full inverse-diagonal scaling was also worse than the square-root form.

## Rejected v20 alternatives

- Pure cosine rejection initially improved convergence but later starved the history and increased line-search work.
- Redundancy replacement produced large 5k gains but did not survive the 20k 0.001% gate consistently.
- Curvature-angle eviction damaged the chronological secant sequence.
- Periodic and direction-quality-triggered replacement remained trajectory-sensitive.
- Two-timescale downsampling and late history 192 did not improve the retained baseline.
- Dense BFGS continuation below 0.005% produced only a small improvement, below the release bar.
- Full inverse Gauss-Newton diagonal scaling was less robust than the square-root transform.

## Numerical contract

The Gauss-Newton diagonal is positive by construction. Median normalisation makes the scale dimensionless, clipping bounds the coordinate correction, and v18's curvature safeguard remains authoritative for every stored pair. If the median diagonal is non-positive or non-finite, v20 falls back to an all-ones diagonal and therefore to v19 scaling.
