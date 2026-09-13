# v17: output-layer SIMD and deterministic parallel confirmation

V17 optimises fixed overheads that became material after v16 reduced the L-BFGS solve to three very fast evaluations.

## Retained changes

1. The eight output pre-activations in each AVX-512 row group are passed through one FP64 vector sigmoid instead of eight scalar sigmoids.
2. Squared error and percentage error are accumulated vectorially for those eight observations.
3. Final stopping confirmation still calls the inherited scalar `predictRow` for every observation, but predictions are computed across up to four workers. Their errors are then aggregated in original row order.
4. L-BFGS, Armijo line search, FP64 weights/gradients, four-worker cap and the 3.9% stopping target are unchanged.
5. Unsupported builds fall back to v16.

## End-to-end time to target

Seven alternating paired runs, including the final confirmation pass:

| Rows | v16 median | v17 median | Reduction from medians | Median paired reduction |
|---:|---:|---:|---:|---:|
| 100,000 | 0.02605 s | 0.01390 s | 46.6% | 45.3% |
| 500,000 | 0.11190 s | 0.05981 s | 46.6% | 49.8% |
| 1,000,000 | 0.21655 s | 0.08421 s | 61.1% | 60.5% |

Both versions reached target in exactly three L-BFGS evaluations in every retained run. Final confirmed percentage error differed by at most 2.22e-15 across the release matrix.

## Rejected experiments

- Persistent `std::thread` worker pool: slower at 100k and 500k and not robustly better at 1m. OpenMP already reuses workers efficiently.
- SoA input copy: did not repay its extra addressing/storage cost for this 11-feature network.
- Metrics-only line-search candidate evaluation: slower because forward sigmoid work dominates, so an extra metrics pass costs too much.
- Skipping the final gradient: same issue. The saved gradient work did not offset the extra forward pass.

## Numerical contract

V17 remains FP64 but is not bit-identical internally to v16 because the grouped output sigmoid and SIMD reductions reassociate floating-point work. Final stopping confirmation uses the inherited scalar `predictRow` implementation for each observation and deterministic row-ordered aggregation.
