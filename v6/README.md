# Simple Neural Network v6

V6 keeps the same 11 -> 16 -> 1 sigmoid network and training equations as v5. It is a smaller, profile-led optimisation round: v5 had already moved most arithmetic into specialised AVX2/AVX-512 kernels, so v6 focuses on reducing scratch-memory traffic and avoiding diagnostics that are not needed on every descent.

As with the earlier versions, the complete implementation remains in one `main.cpp`.

## What changed from v5

- Retuned the training block from 64 rows to 16 after a block-size sweep on the benchmark machine.
- Fused the AVX-512 hidden-delta (`deltaTwo`) calculation directly into all 11 W1 gradient accumulators. The fast path no longer writes the 16 hidden deltas back to the block buffer and rereads them once per input feature.
- Ordinary gradient descents calculate the mean percentage error required by the stopping rule, but skip cost and maximum-percentage-error accumulation when those values will not be reported.
- Full cost and maximum-error diagnostics are still calculated at logging points, at the update limit and when training terminates. If the percentage-error target is first reached on a lightweight pass, v6 performs one full metrics pass before reporting the result.
- The AVX2/FMA and portable scalar fallbacks remain available. The optional glibc `libmvec` vector-sigmoid backend is unchanged.

A precomputed percentage-error reciprocal was also tested and rejected: it saved only a small amount of CPU time while adding about 8 MiB at one million rows. An AMD AOCL-LibM backend was investigated but is not included because the package could not be benchmarked in this environment without accepting its download EULA.

## Building

Portable build:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v6/main.cpp -o simple_nn_v6
```

OpenMP + glibc vector-exp build used for the paired benchmark:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    -DSIMPLE_NN_USE_LIBMVEC v6/main.cpp -lm -o simple_nn_v6
```

The arithmetic kernel is selected at runtime: AVX-512/FMA first, then AVX2/FMA, then the portable scalar implementation.

## Paired v5 versus v6 benchmark

The table below is from a fresh paired run of the committed v5 benchmark kernel and the final v6 kernel. Both used identical deterministic data, starting weights, update counts and compiler settings. Each cell is the median of seven repetitions.

| Rows | Updates | v5 single | v6 single | Speed-up | v5 4-thread | v6 4-thread | Speed-up |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13,853 | 500 | 0.275 s | 0.257 s | **1.07x** | 0.087 s | 0.078 s | **1.12x** |
| 100,000 | 100 | 0.407 s | 0.383 s | **1.06x** | 0.133 s | 0.121 s | **1.10x** |
| 500,000 | 20 | 0.438 s | 0.389 s | **1.12x** | 0.122 s | 0.107 s | **1.14x** |
| 1,000,000 | 10 | 0.437 s | 0.385 s | **1.13x** | 0.165 s | 0.123 s | **1.35x** |

V6 is about **6-12% faster than v5 single-threaded** across these workloads. The four-thread improvement ranges from about **9% to 26%**, with the largest gain on the one-million-row case.

Peak resident memory on the 1,000,000-row, four-thread case was 95,488 KiB for v5 and 95,760 KiB for v6, effectively unchanged.

## Progress across versions

The values below combine the recorded benchmark rounds for each version to show historical progress. They are not one paired run, so they should be read as progress tracking rather than precision cross-version benchmarking.

| Rows | Updates | v1 | v2 | v3 single | v4 single | v5 single | v6 single | v6 vs v1 | v6 4-thread | v6 4-thread vs v1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13,853 | 500 | 2.289 s | 1.172 s | 1.091 s | 0.581 s | 0.275 s | 0.257 s | **8.90x** | 0.078 s | **29.35x** |
| 100,000 | 100 | 3.737 s | 1.682 s | 1.490 s | 0.876 s | 0.407 s | 0.383 s | **9.76x** | 0.121 s | **30.92x** |
| 500,000 | 20 | 4.838 s | 1.782 s | 1.515 s | 0.902 s | 0.438 s | 0.389 s | **12.42x** | 0.107 s | **45.40x** |
| 1,000,000 | 10 | 4.975 s | 1.830 s | 1.525 s | 0.862 s | 0.437 s | 0.385 s | **12.92x** | 0.123 s | **40.53x** |

On this historical-progress basis, v6 is roughly **8.90x to 12.92x faster than v1 single-threaded**, and roughly **29.35x to 45.40x faster with four threads**.

## Numerical validation

A production-path 100,000-row, one-update comparison produced byte-identical `wone.txt`, `wtwo.txt` and `ybar.txt` between v5 and v6 at stored precision. The AVX2 v6 path was also forced separately and produced the same hashes.

The paired benchmark finished on the same printed final-weight checksum for v5 and v6 in every workload and thread configuration.

Full processed results, raw timings, the pre-v6 hotspot profile, memory measurement and output hashes are under [`benchmark/`](benchmark/).
