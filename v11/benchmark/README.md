# V11 benchmark notes

This directory records the final v10-style versus v11 adaptive-sigmoid benchmark, numerical-drift checks and rejected v11 experiments.

## Environment

- AMD EPYC 9V74
- GCC 14.2.0
- Linux x86-64
- AVX-512
- glibc libmvec for the exact baseline and v11 fallback
- OpenMP for the four-thread case

The current execution host differs from the historical Xeon host used for v8-v10. Absolute seconds should therefore not be compared across version READMEs. Every v11 result is a paired same-host comparison against a v10-style exact-sigmoid baseline.

## Final approximation

For each eight-lane vector:

- all lanes inside `[-1,1]`: degree-5 odd polynomial;
- any lane outside `[-1,1]`: existing checked libmvec sigmoid for that vector.

This keeps the common path short while retaining exact historical behaviour outside the approximation interval.

Dense-grid maximum absolute error on `[-1,1]` is about `2.69e-6`.

## Release-development cells

Alternating paired repetitions were used:

- 100,000 rows x 100 updates, 1 thread, 7 pairs
- 500,000 rows x 40 updates, 1 thread, 7 pairs
- 1,000,000 rows x 30 updates, 1 thread, 7 pairs
- 1,000,000 rows x 60 updates, 4 threads, 5 pairs

One-thread runs use the four-row production-style forward grouping. The four-thread million-row cell uses the eight-row v10 grouping.

`benchmark_summary.csv` contains medians and paired reductions. `raw_timings.csv` retains every timing pair, maximum final-weight difference and exact-sigmoid RMSE for both trained networks.

## Accuracy and long-run checks

`accuracy.txt` records:

- dense-grid approximation error;
- release-cell final-weight drift and exact-sigmoid RMSE drift;
- 1,000-update single-thread and 300-update four-thread stability checks;
- exact target-confirmation semantics.

## Benchmark source

`bench_v11.cpp` is the self-contained development harness used to compare the production-style exact and adaptive sigmoid kernels. It does not modify historical version sources.
