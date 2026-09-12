# Simple Neural Network v5

V5 keeps the original 11 -> 16 -> 1 sigmoid network and v4 training equations, but specialises the fixed-size arithmetic that became the dominant v4 hotspot. The complete implementation remains in one `main.cpp`.

## Main changes

- Adds runtime-dispatched AVX2/FMA and AVX-512/FMA training kernels while retaining the portable scalar v4 kernel.
- Keeps all 16 hidden forward accumulators in vector registers for each observation instead of repeatedly updating memory.
- Keeps each 16-value W1 gradient row in vector registers across a full block, storing it only after the block has been accumulated.
- Uses two independent gradient accumulator chains to reduce FMA dependency latency without the register pressure seen with a four-chain experiment.
- Vectorises the W2 gradient/delta calculation and the 16-wide output dot product.
- Retains v4's 64-row blocking, vector sigmoid backend and persistent OpenMP worker team.

## Building

Portable C++11 build:
```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v5/main.cpp -o simple_nn_v5
```

OpenMP build:
```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp v5/main.cpp -o simple_nn_v5
```

Accelerated build used for the benchmark:
```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    -DSIMPLE_NN_USE_LIBMVEC v5/main.cpp -lm -o simple_nn_v5
```

The arithmetic kernel is selected at runtime: AVX-512/FMA first, then AVX2/FMA, then the portable scalar fallback. `SIMPLE_NN_USE_LIBMVEC` controls the optional glibc vector-`exp()` sigmoid backend separately, so the whole program does not need a global `-march=native` build.

## Paired v4 versus v5 benchmark

These medians were measured in the same benchmark round with identical deterministic data, starting weights, update counts and compiler settings. Dataset generation and file I/O are outside the timed training region. Seven repetitions were used for the first three workloads and five for the 1,000,000-row workload.

| Rows | Updates | v4 single | v5 single | Speed-up | v4 4-thread | v5 4-thread | Speed-up |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13,853 | 500 | 0.576 s | 0.274 s | **2.10x** | 0.173 s | 0.091 s | **1.89x** |
| 100,000 | 100 | 0.845 s | 0.402 s | **2.10x** | 0.243 s | 0.121 s | **2.00x** |
| 500,000 | 20 | 0.852 s | 0.436 s | **1.96x** | 0.347 s | 0.132 s | **2.63x** |
| 1,000,000 | 10 | 0.879 s | 0.448 s | **1.96x** | 0.279 s | 0.168 s | **1.66x** |

Single-thread v5 is about **1.96x to 2.10x faster than v4** across the four workloads. The four-thread gain ranges from about **1.66x to 2.63x**, with more variation because the measured times are now short enough for scheduler noise to matter.

Peak RSS on the 1,000,000-row, four-thread case was 95,616 KiB for v4 and 95,744 KiB for v5, effectively unchanged.

## Progress across versions

The table below combines each version's recorded benchmark round to show cumulative progress. It is a historical progress table rather than a single paired run.

| Rows | Updates | v1 | v2 | v3 single | v4 single | v5 single | v5 vs v1 | v5 4-thread | v5 4-thread vs v1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13,853 | 500 | 2.289 s | 1.172 s | 1.091 s | 0.581 s | 0.274 s | **8.34x** | 0.091 s | **25.10x** |
| 100,000 | 100 | 3.737 s | 1.682 s | 1.490 s | 0.876 s | 0.402 s | **9.30x** | 0.121 s | **30.78x** |
| 500,000 | 20 | 4.838 s | 1.782 s | 1.515 s | 0.902 s | 0.436 s | **11.10x** | 0.132 s | **36.67x** |
| 1,000,000 | 10 | 4.975 s | 1.830 s | 1.525 s | 0.862 s | 0.448 s | **11.12x** | 0.168 s | **29.54x** |

On this progress-tracking basis, v5 is roughly **8.34x to 11.12x faster than v1 single-threaded**, while the four-thread path is roughly **25.10x to 36.67x faster** on the measured workloads.

## Numerical validation

A production-path 100,000-row, one-update comparison produced byte-identical `wone.txt`, `wtwo.txt` and `ybar.txt` between v4 and v5 at stored precision. The AVX2 path was separately forced and produced the same hashes. The controlled benchmark also finished on the same printed weight checksum in every workload. Parallel diagnostics can still differ at floating-point round-off level because worker reductions are grouped differently.

Full benchmark data, raw timings, the pre-v5 profile summary, memory measurement and equivalence hashes are under [`benchmark/`](benchmark/).
