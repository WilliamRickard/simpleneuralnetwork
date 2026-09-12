# Simple Neural Network v7

V7 keeps the same 11 -> 16 -> 1 sigmoid network, double-precision weights and training equations as v6. It is another profile-led CPU optimisation round.

The complete implementation remains in one `main.cpp`.

## What changed from v6

- The AVX-512 forward kernel now processes four observations together. This creates eight independent hidden-layer FMA chains and reuses each loaded W1 vector across four rows, reducing dependency stalls in the 11 x 16 forward multiply.
- The percentage-error and `deltaThree` calculations are vectorised across observations with AVX-512. Percentage-error values are still accumulated in row order afterwards, which preserves the stopping metric's summation order.
- The v6 fused AVX-512 backpropagation and W1 gradient accumulation are retained unchanged.
- The v6 AVX-512 kernel remains available as a fallback. On the benchmark machine, the 13,853-row four-thread workload was too short for the new kernel to give a reliable benefit, so parallel batches below 50,000 rows retain the v6 path.
- The AVX2/FMA and portable scalar fallbacks remain available.
- The optional glibc `libmvec` vector-exp backend is unchanged.

Several ideas were tested but not retained. A four-tile superblock increased register pressure and gave inconsistent results. Rewriting sigmoid as `0.5 * (1 + tanh(x/2))` was substantially slower with glibc `libmvec`. Memory-broadcast FMA was not pursued because each input scalar feeds two hidden-vector FMAs, so replacing one broadcast with two memory-broadcast FMAs would duplicate scalar loads.

## Building

Portable build:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v7/main.cpp -o simple_nn_v7
```

OpenMP + glibc vector-exp build used for the paired benchmark:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    -DSIMPLE_NN_USE_LIBMVEC v7/main.cpp -lm -o simple_nn_v7
```

Runtime arithmetic dispatch is:

1. v7 AVX-512/FMA/DQ kernel when supported and the workload is suitable
2. v6 AVX-512/FMA fallback
3. AVX2/FMA
4. portable scalar C++

## Paired v6 versus v7 benchmark

This benchmark was run on an Intel Xeon Platinum 8573C with five exposed physical cores. V6 and v7 used identical deterministic data, starting weights, update counts and compiler settings. Each cell is the median of nine repetitions, and run order alternated between v6 and v7.

| Rows | Updates | v6 single | v7 single | Speed-up | v6 4-thread | v7 4-thread | Speed-up |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13,853 | 500 | 0.250 s | 0.233 s | **1.07x** | 0.131 s | 0.123 s* | n/a* |
| 100,000 | 100 | 0.369 s | 0.339 s | **1.09x** | 0.189 s | 0.167 s | **1.13x** |
| 500,000 | 20 | 0.381 s | 0.358 s | **1.07x** | 0.188 s | 0.171 s | **1.10x** |
| 1,000,000 | 10 | 0.402 s | 0.369 s | **1.09x** | 0.188 s | 0.169 s | **1.12x** |

`*` The 13,853-row four-thread configuration deliberately dispatches to the unchanged v6 kernel. The small timing difference between the two columns is therefore scheduler/run-order noise, not a v7 algorithmic speed-up.

On workloads that actually use the new AVX-512 kernel, v7 reduced elapsed training time by roughly **6-12%** in this benchmark round.

Peak resident memory for the 1,000,000-row, four-thread, one-update memory test was **95,616 KiB for both v6 and v7**.

## Progress across versions

The v1-v6 historical timings were measured in earlier benchmark rounds, including a different CPU environment. Because v7 was measured on the Xeon environment above, its absolute seconds should not be appended directly to that historical table.

A useful hardware-normalised progress measure is to multiply the historical v6-vs-v1 speed-up by the fresh paired v7-vs-v6 speed-up. On that basis, the single-thread progression is approximately:

| Rows | Historical v6 vs v1 | Paired v7 vs v6 | Normalised v7 vs v1 |
|---:|---:|---:|---:|
| 13,853 | 8.90x | 1.07x | **9.52x** |
| 100,000 | 9.76x | 1.09x | **10.62x** |
| 500,000 | 12.42x | 1.07x | **13.23x** |
| 1,000,000 | 12.92x | 1.09x | **14.06x** |

For four threads, excluding the small-workload fallback, the same normalisation gives approximately **35.0x**, **50.1x** and **45.2x** versus v1 for the 100k, 500k and 1m workloads respectively. These are derived progress ratios, not direct v1-versus-v7 measurements.

## Numerical validation

A production-path 100,000-row, one-update comparison produced byte-identical `wone.txt`, `wtwo.txt` and `ybar.txt` between the retained v6 kernel and the new v7 AVX-512 kernel at stored precision.

The AVX2 path was also forced separately and produced the same output hashes.

The deterministic paired benchmark finished on the same final-weight checksum for v6 and v7 in every workload and thread configuration.

Full processed results, raw timings, the pre-v7 hotspot profile, memory measurement and output hashes are under [`benchmark/`](benchmark/).
