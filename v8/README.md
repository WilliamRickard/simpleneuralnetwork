# Simple Neural Network v8

V8 keeps the same 11 -> 16 -> 1 sigmoid network, double-precision weights and training equations as v7. It is a conservative exact-model optimisation round: changes are enabled only on workload shapes where paired benchmarking showed a useful result.

The complete implementation remains in one `main.cpp`.

## What changed from v7

- Single-thread batches of at least 50,000 rows use a range-level AVX-512 driver rather than dispatching one external training-kernel call for every 16 rows.
- The single-thread v8 tile no longer zero-initialises the 2 KiB hidden scratch array. The forward microkernel overwrites every used activation before it is read, so those stores were redundant.
- The vector sigmoid backend is resolved once and cached on the v8 paths instead of repeating CPU-feature tests for every sigmoid call.
- On lightweight single-thread passes, percentage-error divisions stop once the accumulated non-negative error is already above `percentageErrorTarget * rowCount`. At that point the stopping target is mathematically impossible for that pass. Forward propagation, backpropagation and every gradient calculation continue unchanged.
- Detailed/logging passes always calculate the full cost, mean percentage error and maximum percentage error.
- Batches below 50,000 rows deliberately retain the v7 path because the extra range machinery did not help the 13,853-row benchmark.
- Parallel batches below 50,000 rows retain the v6 fallback, as in v7. Parallel batches from 50,000 to below 1,000,000 rows retain the v7 kernel. At 1,000,000 rows and above, v8 keeps the v7 arithmetic but caches vector-sigmoid dispatch.
- AVX2/FMA and portable scalar fallbacks remain available.

The percentage cutoff does **not** approximate the training rule. Percentage errors are non-negative, so once the partial sum exceeds the full-pass target sum, later rows cannot make the final mean fall back below the target. If the target can be reached, v8 never crosses the cutoff and therefore computes the full percentage sum.

Third-party vector-math backends were researched but not added because oneMKL, SLEEF and AOCL-LibM are not installed in this environment. The optional polynomial sigmoid experiment also remains outside the exact/default v8 path.

## Building

Portable build:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v8/main.cpp -o simple_nn_v8
```

OpenMP + glibc vector-exp build used for the paired benchmark:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    -DSIMPLE_NN_USE_LIBMVEC v8/main.cpp -lm -o simple_nn_v8
```

Both builds compile cleanly with GCC 14.2.0 and the warning flags above.

## Paired v7 versus v8 benchmark

The benchmark was run on an Intel Xeon Platinum 8573C with five exposed physical cores. V7 and v8 used identical deterministic data, starting weights, update counts and compiler settings. Each cell is the median of nine repetitions and run order alternated between v7 and v8. One-thread runs were pinned to CPU 0. Four-thread runs were pinned to CPUs 0-3 with `OMP_PROC_BIND=close` and `OMP_PLACES=cores`.

| Rows | Updates | v7 single | v8 single | Speed-up | v7 4-thread | v8 4-thread | Speed-up |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13,853 | 500 | 0.322 s | 0.338 s* | n/a* | 0.126 s | 0.136 s* | n/a* |
| 100,000 | 100 | 0.367 s | 0.343 s | **1.07x** | 0.181 s | 0.181 s* | n/a* |
| 500,000 | 20 | 0.356 s | 0.332 s | **1.07x** | 0.170 s | 0.170 s* | n/a* |
| 1,000,000 | 10 | 0.369 s | 0.352 s | **1.05x** | 0.176 s | 0.169 s | **1.04x** |

`*` These configurations deliberately dispatch to the same training kernel in v7 and v8. Their measured timing differences are scheduler/run-order noise and are not claimed as v8 gains or regressions.

On the workloads that actually use the new single-thread range kernel, v8 reduced elapsed training time by about **5-7%** in this final paired sweep. The one-million-row four-thread cached-dispatch path was about **4% faster**.

Peak resident memory for the 1,000,000-row, four-thread, one-update memory test was 95,488 KiB for v7 and 95,616 KiB for v8, effectively unchanged.

## Pre-v8 profile

The median sampled v7 CPU-time shares on 1,000,000 rows x 20 updates were:

| Stage | 1 thread | 4 threads |
|---|---:|---:|
| Hidden sigmoid | 37.0% | 43.7% |
| Forward 11 x 16 multiply | 20.4% | 15.3% |
| Backprop + gradients | 18.7% | 19.9% |
| Metric + `deltaThree` | 8.8% | 7.1% |
| Output sigmoid | 8.2% | 5.6% |
| Output dot product | 6.5% | 6.3% |

Sigmoid therefore accounts for roughly 45-49% of sampled CPU cycles. V8 removes overhead around that work without changing the sigmoid implementation itself.

## Progress across versions

V1-v6 historical timings include an earlier AMD EPYC benchmark environment, while v7 and v8 were measured on the Xeon environment above. Absolute seconds should therefore not be appended directly into one long timing table.

Using the same hardware-normalised method as the v7 README, multiply the previous normalised v7-vs-v1 ratio by the fresh paired v8-vs-v7 ratio only where v8 actually changes the kernel:

| Rows | Normalised v7 vs v1 | Paired v8 vs v7 | Normalised v8 vs v1 |
|---:|---:|---:|---:|
| 13,853 | 9.52x | unchanged | **9.52x** |
| 100,000 | 10.62x | 1.07x | **11.35x** |
| 500,000 | 13.23x | 1.07x | **14.21x** |
| 1,000,000 | 14.06x | 1.05x | **14.78x** |

For four threads, the normalised progress is unchanged at about **29.35x**, **35.0x** and **50.1x** versus v1 for the 13,853, 100k and 500k workloads respectively. The one-million-row cached-dispatch path increases the previous normalised 45.2x figure to about **47.1x**. These are derived progress ratios, not direct v1-versus-v8 measurements.

## Numerical validation

A production-path 100,000-row one-update comparison was run separately with one thread and four threads. In both cases v7 and v8 produced byte-identical `wone.txt`, `wtwo.txt` and `ybar.txt` at stored precision.

The deterministic paired benchmark finished on the same final-weight checksum for v7 and v8 in every workload and thread configuration.

Full processed results, raw timings, the pre-v8 hotspot profile, memory measurement and output hashes are under [`benchmark/`](benchmark/).
