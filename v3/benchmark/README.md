# V3 benchmark

This directory records the controlled v2 versus v3 performance comparison.

The benchmark keeps the model fixed at 11 inputs, 16 sigmoid hidden nodes and one sigmoid output. Both versions use identical deterministic synthetic observations, starting weights, learning rate, momentum and update counts. Data generation, console logging and file I/O are outside the timed training region.

The v2 baseline is the benchmark harness already stored under `v2/benchmark/`. The v3 harnesses reproduce the v3 fused training kernel in single-threaded and OpenMP forms.

## Compiler

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic
```

The OpenMP harness adds:

```text
-fopenmp
```

The compiler already auto-vectorised the small arithmetic loops at `-O3`. Additional `-march=native`, `-fno-math-errno`, `-ffast-math` and PGO experiments did not show a sufficiently consistent improvement to retain.

## Results

See `benchmark_summary.csv` for processed medians and `raw_timings.csv` for the individual runs.

The headline result on 1,000,000 rows is:

```text
v2:                 1.804 s
v3 single-thread:   1.525 s   1.18x faster
v3 four-thread:     0.773 s   2.33x faster than v2
```

### Progress from v1

The earlier paired v1-versus-v2 benchmark under `v2/benchmark/` recorded a 1,000,000-row v1 median of 4.975 s. Compared with that original baseline:

```text
v1 baseline:        4.975 s
v3 single-thread:   1.525 s   3.26x faster than v1
v3 four-thread:     0.773 s   6.44x faster than v1
```

Across all four workloads, v3 single-thread is about 2.10x to 3.26x faster than the original v1 baseline, while the four-thread path is about 4.88x to 6.92x faster. These cumulative comparisons combine the original v1 baseline round with the later v3 measurements, so the paired v2-versus-v3 figures remain the stricter like-for-like comparison.

Peak resident memory for a one-update 1,000,000-row process was:

```text
v2:   228,224 KiB  (~223 MiB)
v3:    95,488 KiB  (~93 MiB)
```

That is a further memory reduction of about 58% from v2 to v3.

## Numerical checks

The single-thread v3 harness produced the same final weight checksum as v2 in every benchmark case. The four-thread reduction changes summation order, so aggregate cost and percentage-error values can differ at floating-point round-off level. The largest observed cost difference in these runs was approximately `1.4e-10`, while the reported final weight checksum remained the same.

## Environment

The benchmark execution environment exposed five CPU cores from an AMD EPYC 9V74 processor, with one hardware thread per exposed core. Four OpenMP worker threads gave the best stable timings in the measured cases.
