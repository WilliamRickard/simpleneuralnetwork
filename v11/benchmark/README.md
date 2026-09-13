# V11 benchmark notes

This directory records the v10-style versus v11 sigmoid-isolation benchmark, numerical-drift checks and rejected v11 experiments.

## Environment

- Intel Xeon Platinum 8573C
- GCC 14.2.0
- Linux x86-64
- AVX-512
- glibc libmvec for the exact baseline sigmoid
- OpenMP for the four-thread case

oneMKL and SLEEF were not installed in the benchmark environment, so v11 does not add either as an untested dependency. Current oneMKL provides vector exponential routines with multiple accuracy modes, but that comparison is left for a future dependency-enabled benchmark.

## Release-development cells

Five alternating paired repetitions were used for each cell:

- 100,000 rows x 300 updates, 1 thread
- 500,000 rows x 60 updates, 1 thread
- 1,000,000 rows x 30 updates, 1 thread
- 1,000,000 rows x 100 updates, 4 threads

One-thread runs use the v9-style four-row forward grouping. The four-thread million-row cell uses the v10 eight-row grouping. In each pair the only intended hot-kernel change is exact libmvec sigmoid versus the v11 adaptive polynomial sigmoid.

`benchmark_summary.csv` contains medians and paired reductions. `raw_timings.csv` retains every timing pair and the maximum final-weight difference.

## Accuracy and long-run checks

`accuracy.txt` records:

- dense-grid sigmoid approximation error
- observed preactivation ranges
- exact-sigmoid RMSE and max prediction error after training
- long-run 1,000-update and 300-update drift tests
- exact target-confirmation semantics

## Benchmark source

`bench_v11.cpp` is the exact local development harness used for the reported paired measurements. It is intentionally self-contained so the approximation and the v10-style baseline can be reproduced without altering historical version sources. It requires C++17 because its driver uses generic lambdas; production `v11/main.cpp` remains C++11.

Example:

```text
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -fopenmp bench_v11.cpp -lm -o bench_v11
OMP_PROC_BIND=close OMP_PLACES=cores taskset -c 0-3 ./bench_v11 1000000 100 5 4
```
