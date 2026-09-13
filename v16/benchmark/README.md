# v16 benchmark

The benchmark compares v15's four-thread scalar FP64 evaluator with v16's four-thread AVX-512 FP64 evaluator inside the same L-BFGS solve.

Use the committed harness with seven alternating paired runs, for example:

```bash
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp -march=native v16/benchmark/bench_v16.cpp -lm -o bench_v16
OMP_PROC_BIND=true OMP_PLACES=cores ./bench_v16 1000000 7
```

Release summary and raw timings are in `benchmark_summary.csv` and `raw_timings.csv`.
