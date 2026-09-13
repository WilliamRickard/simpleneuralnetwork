# V15 benchmark notes

V15 measures end-to-end time to the existing 3.9% percentage-error target for the v14 L-BFGS algorithm.

The retained comparison changes only evaluator parallelism:

- baseline: one exact FP64 evaluator worker;
- candidate: four exact FP64 evaluator workers.

Each release cell uses seven alternating paired repetitions to reduce order bias. Both implementations start from identical deterministic weights and use the same dataset, objective, line search and L-BFGS history rules.

Release cells:

- 100,000 rows;
- 500,000 rows;
- 1,000,000 rows.

Every retained run reached the target on the third exact objective/gradient evaluation.

`benchmark_summary.csv` contains median times and paired reductions. `raw_timings.csv` contains every retained timing pair.

The harness also contains the rejected FP32-proposal evaluator used during screening. That path was not used for release timings because adding proposal passes made end-to-end solve time worse once exact Armijo acceptance was retained.

Build:

```text
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    v15/benchmark/bench_v15.cpp -o bench_v15
```

For consistent affinity on Linux, release measurements used `OMP_PROC_BIND=close` and `OMP_PLACES=cores`.
