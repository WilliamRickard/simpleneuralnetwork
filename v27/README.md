# v27: exact parallel Gauss-Newton refresh

V27 preserves v26's optimiser, evaluator, dense BFGS algebra, line search and stopping behaviour. It accelerates only the Gauss-Newton diagonal refresh used before the 0.015% dense handover on deep targets.

## Why this exists

V25 replaced the stale sub-50k forced-serial policy for the main evaluator, but the inherited `gaussNewtonScaleV20` still calls `evaluationThreadsV15`. That leaves each Gauss-Newton scale pass serial below 50,000 rows even though the surrounding v25 evaluator already uses adaptive parallelism.

Simply switching the v20 accumulation to multiple partial sums changes floating-point grouping and changes the optimiser trajectory. That candidate was rejected.

## Retained implementation

For batches below 50,000 rows and targets at or below 0.001%, v27 separates each Gauss-Newton refresh into two phases:

1. prepare each row's 192 Jacobian doubles in parallel;
2. square and accumulate those values into the inherited `long double` diagonal serially in the original row order.

The row preparation uses AVX-512 across the 16 hidden nodes. Each hidden-node lane still accumulates the 11 inputs in `k = 0..10` order. Multiplication and addition are separate and function-local `fp-contract=off` prevents contraction. Scalar `exp` remains unchanged.

The scratch buffer is bounded to 8,192 rows, or about 12.6 MB at 192 doubles per row, and exists only for the duration of one Gauss-Newton refresh.

V27 delegates directly to v26 when:

- the target is above 0.001%;
- the batch has 50,000 rows or more;
- the accelerated/libmvec path is unavailable; or
- runtime AVX-512 support is unavailable.

## Exactness

The focused benchmark compares the original scalar diagonal against the v27 block-parallel implementation bit-for-bit.

The production-style optimiser harness then compared the paths at 5k, 10k and 20k rows at both 0.001% and 0.0005%. Every case had identical accepted updates, evaluator calls, rejected line-search trials, final percentage error, final objective and final parameter bits.

The real 13.3k Wine stress workload also retained identical final parameters, hold-out RMSE and classification accuracy at both deep targets.

At 50k and 100k the production wrapper delegates directly to v26, so those workloads are unchanged by construction.

## Performance

Rotating-order end-to-end timings are recorded in `benchmark/benchmark_summary.csv`. The benchmark host has a four-CPU cgroup quota, so full-training timing is noisy and the distribution is reported rather than cherry-picking individual runs.

The 0.001% synthetic median speedups were approximately 1.22x at 5k, 1.03x at 10k and 1.08x at 20k. The longer 0.0005% runs amortise the fixed Gauss-Newton saving and were approximately neutral to 1.01x in the synthetic harness.

On the 13.3k real Wine stress workload, ten paired runs gave median speedups of approximately 1.03x at 0.001% and 1.03x at 0.0005%, with identical hold-out quality.

These are direct end-to-end measurements on the benchmark host. They should not be interpreted as universal hardware speedups.

## Validation

The release gate requires the exact production wrapper to compile under:

```text
-O3 -Wall -Wextra -Wpedantic -Werror
-O3 -Wall -Wextra -Wpedantic -Werror -fopenmp -DSIMPLE_NN_USE_LIBMVEC -lmvec
```

The GitHub Actions result must be checked separately. A local or standalone harness result is not a CI-pass claim.
