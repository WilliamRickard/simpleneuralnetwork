# V8 benchmark notes

This directory records the controlled v7-versus-v8 benchmark and validation used for the v8 README.

## Environment

- Intel Xeon Platinum 8573C
- 5 exposed physical cores, one hardware thread per exposed core
- GCC 14.2.0
- Linux x86-64
- glibc `libmvec` vector math
- no oneMKL, SLEEF or AOCL-LibM installed

Accelerated build:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    bench_v8.cpp -lm -o bench_v8
```

The harness contains the committed v7 deterministic baseline and the final v8 dispatch shape. Each workload/thread cell was run nine times with alternating v7/v8 order. Single-thread runs were pinned with `taskset -c 0`; four-thread runs used `taskset -c 0-3`, `OMP_PROC_BIND=close` and `OMP_PLACES=cores`.

## Dispatch represented by the benchmark

- single thread, <50,000 rows: unchanged v7 kernel
- single thread, >=50,000 rows: v8 range driver with cached sigmoid dispatch, uninitialised overwritten scratch, and mathematically safe percentage-error cutoff
- four threads, <50,000 rows: unchanged v6 fallback used by v7
- four threads, 50,000 to <1,000,000 rows: unchanged v7 kernel
- four threads, >=1,000,000 rows: v7 arithmetic with cached sigmoid dispatch

Rows marked as fallback in `benchmark_summary.csv` intentionally execute the same arithmetic/kernel as v7. Any measured difference in those rows is benchmark noise, not an algorithmic v8 change.

## Files

- `bench_v8.cpp`: deterministic paired benchmark harness
- `benchmark_summary.csv`: medians, standard deviations, dispatch path and checksum comparison
- `raw_timings.csv`: all nine paired repetitions for every workload/thread configuration
- `profile_before_v8.txt`: seven-run sampled v7 stage profile that motivated v8
- `memory.txt`: one-million-row four-thread peak RSS comparison
- `equivalence.txt`: production text-file one-update equivalence results and hashes

## Experiments retained and rejected

Retained:

- remove hidden scratch zero-fill on the single-thread v8 path
- cache vector-sigmoid backend selection
- process the single-thread workload through one range driver
- after the target becomes mathematically impossible on a lightweight pass, switch to a specialised no-percentage-division tile
- cache sigmoid dispatch on very large parallel batches

Not retained:

- a branch inside every metric tile to decide whether to calculate percentage error: slower than separate specialised tile paths
- applying the uninitialised-scratch/range dataflow to all OpenMP workloads: not consistently faster
- forcing AVX2 sigmoid on the AVX-512 machine: slower in the profiling experiments
- `tanh` sigmoid identity: slower with glibc vector math
- `exp2(-x * log2(e))`: interesting single-thread results but not consistently better in parallel
- approximate polynomial sigmoid: large experimental speed-up, but deliberately excluded from exact/default v8
