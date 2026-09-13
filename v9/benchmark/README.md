# V9 benchmark notes

This directory records the controlled v8-versus-v9 benchmark and validation used for the v9 README.

## Environment

- Intel Xeon Platinum 8573C
- 5 exposed physical cores, one hardware thread per exposed core
- GCC 14.2.0
- Linux x86-64
- glibc `libmvec` vector math
- no oneMKL, SLEEF or AOCL-LibM installed

Benchmark build, run from this directory:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp bench_v9.cpp -lm -o bench_v9
```

`bench_v9.cpp` includes the frozen v8 benchmark harness from `../../v8/benchmark/bench_v8.cpp` and adds only the v9 comparison path. One-thread runs were pinned with `taskset -c 0`. The four-thread run used `taskset -c 0-3`, `OMP_PROC_BIND=close` and `OMP_PLACES=cores`.

## Release benchmark method

The release summary includes only configurations that actually dispatch to v9. Short screening runs showed enough host jitter that sub-second cells could move by several percentage points, so the final active-path measurements use longer runs:

- 100,000 rows x 300 updates, 1 thread
- 500,000 rows x 60 updates, 1 thread
- 1,000,000 rows x 30 updates, 1 thread
- 1,000,000 rows x 30 updates, 4 threads

Each cell contains nine paired repetitions. V8 and v9 execute in the same process and the run order alternates by repetition. `benchmark_summary.csv` contains medians, standard deviations and the maximum checksum difference. `raw_timings.csv` retains every pair.

## Dispatch represented by the benchmark

- one thread, below 50,000 rows: frozen v8 fallback
- one thread, at least 50,000 rows: v9 full-tile/range path with the conservative unchecked-sigmoid guard
- four threads, below 1,000,000 rows: frozen v8 fallback
- four threads, at least 1,000,000 rows: v9 contiguous range path with the same guarded full-tile kernel

Fallback configurations were tested during screening but are omitted from the release performance table because v9 intentionally does not change their training kernel.

## Exactness validation

All 36 release benchmark pairs finished with zero final checksum difference.

`equivalence.txt` records the stronger independent check: complete W1, W2, deltaW1 and deltaW2 arrays are byte-identical for the 100,000-row one-thread path and 1,000,000-row four-thread path.

## Memory

`memory.txt` records the one-million-row, four-thread peak-RSS comparison. V8 used 95,616 KiB and v9 used 95,624 KiB.

## Experiments retained and rejected

Retained:

- full 16-row AVX-512 tile specialisation
- direct AVX-512 sigmoid invocation inside the specialised tile
- per-dataset feature maxima and a conservative per-pass pre-activation bound
- unchecked libmvec sigmoid only when that bound proves the `-700` exceptional branch unreachable
- v9 parallel range driver only from 1,000,000 rows

Rejected:

- 64-byte aligning `Network` and converting hot weight reads to aligned loads: inconsistent benefit
- direct cached-dispatch removal by itself: below the noise floor
- explicit OpenMP block ownership by itself: too small to retain
- carrying the 24-vector W1/W2 gradient state across tiles: generated assembly already showed substantial ZMM spills, making additional live state unattractive
- oneMKL/SLEEF/AOCL-LibM comparisons: libraries unavailable in the benchmark environment

The source model and training equations are unchanged from v8.
