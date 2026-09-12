# V5 benchmark

This directory records the controlled v4-versus-v5 benchmark and the profile evidence that motivated v5.

## Why v5 targets arithmetic

After v4 vectorised sigmoid evaluation, profiling the accelerated four-thread path showed that W1/W2 backpropagation and gradient accumulation accounted for about 41.7% of CPU time and the hidden forward multiply about 22.8%. Vector sigmoid/`exp()` had fallen to about 13%. V5 therefore specialises the fixed 11 x 16 multiply-add kernels rather than changing the model or sigmoid again.

## Method

The workloads use the same deterministic synthetic 11-input observations and starting weights as the earlier benchmarks. Training time excludes data generation and file I/O. V4 and v5 use the same learning rate, momentum and full-batch update count.

| Rows | Updates |
|---:|---:|
| 13,853 | 500 |
| 100,000 | 100 |
| 500,000 | 20 |
| 1,000,000 | 10 |

The benchmark machine exposed five AMD EPYC 9V74 cores with AVX2, FMA and AVX-512. The accelerated sigmoid build used glibc `libmvec`.

## Results

These medians were measured in the same benchmark round with identical deterministic data, starting weights, update counts and compiler settings. Dataset generation and file I/O are outside the timed training region. Seven repetitions were used for the first three workloads and five for the 1,000,000-row workload.

| Rows | Updates | v4 single | v5 single | Speed-up | v4 4-thread | v5 4-thread | Speed-up |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13,853 | 500 | 0.576 s | 0.274 s | **2.10x** | 0.173 s | 0.091 s | **1.89x** |
| 100,000 | 100 | 0.845 s | 0.402 s | **2.10x** | 0.243 s | 0.121 s | **2.00x** |
| 500,000 | 20 | 0.852 s | 0.436 s | **1.96x** | 0.347 s | 0.132 s | **2.63x** |
| 1,000,000 | 10 | 0.879 s | 0.448 s | **1.96x** | 0.279 s | 0.168 s | **1.66x** |

Single-thread v5 is about **1.96x to 2.10x faster than v4** across the four workloads. The four-thread gain ranges from about **1.66x to 2.63x**, with more variation because the measured times are now short enough for scheduler noise to matter.

Peak RSS on the 1,000,000-row, four-thread case was 95,616 KiB for v4 and 95,744 KiB for v5, effectively unchanged.

Exact processed values are in `benchmark_summary.csv` and individual measurements are in `raw_timings.csv`.

## Correctness and portability

`equivalence.txt` records the byte-identical production output hashes. The AVX2 kernel was forced on the AVX-512 benchmark machine and passed the same check. `main.cpp` selects AVX-512/FMA, AVX2/FMA or scalar arithmetic at runtime, so the accelerated kernels do not require a global CPU-specific executable.

`bench_v5.cpp` is the compact controlled harness for the AVX-512 path used on the benchmark machine. The production source contains the complete runtime-dispatch implementation.
