# Simple Neural Network v9

V9 keeps the same 11 -> 16 -> 1 sigmoid network, double-precision weights and training equations as v8. It is an exact-model optimisation round focused on the AVX-512/libmvec path. V9 deliberately reuses the frozen v8 implementation for fallbacks and layers the specialised training path on top of it.

## What changed from v8

- Complete 16-row AVX-512 tiles use a dedicated full-tile kernel. The rare final partial tile continues through the v8 tile.
- The v9 full-tile path calls the AVX-512 vector sigmoid directly instead of passing through the cached function-pointer wrapper.
- Dataset loading records the maximum absolute value of each input feature.
- Before each training pass, v9 computes a conservative long-double bound on every hidden pre-activation and on the minimum output pre-activation. When the bound proves every relevant value is strictly inside +/-699, v9 uses an unchecked libmvec sigmoid loop that omits the repeated per-vector `x < -700` guard. If the proof fails, the checked v8 sigmoid path is used automatically.
- Single-thread batches of at least 50,000 rows use the v9 range driver on supported AVX-512/libmvec builds.
- Parallel batches use the new range path only from 1,000,000 rows. Smaller parallel batches retain the v8 dispatch because they did not show a robust benefit.
- Non-x86, non-GNU, non-libmvec and unsupported-CPU configurations delegate to v8.

The unchecked sigmoid path changes no approximation or training equation. It calls the same glibc vector `exp` symbol used by v8 and removes only a branch whose condition has first been proven unreachable for that pass. A one-unit safety margin is kept between the proof bound and v8's `-700` fallback threshold.

## Source layout

`v9/main.cpp` includes `../v8/main.cpp` after renaming v8's top-level entry points, then defines the v9-specific training path. This keeps the already-validated v8 fallback frozen instead of duplicating it. Build from the repository root so the relative include resolves normally.

## Building

Portable build:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v9/main.cpp -o simple_nn_v9
```

OpenMP + glibc vector-exp build used for benchmarking:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    -DSIMPLE_NN_USE_LIBMVEC v9/main.cpp -lm -o simple_nn_v9
```

The v9-specific source was compile-checked in both modes with GCC 14.2.0. The performance harness was compiled and run on the same Intel Xeon Platinum 8573C environment used for v8.

## Paired v8 versus v9 benchmark

The release benchmark times only configurations that actually dispatch to v9. To reduce host jitter, the active cells use roughly three times the update counts used in the short screening runs. V8 and v9 execute in the same process with alternating order for nine paired repetitions. One-thread runs were pinned to CPU 0. The four-thread run was pinned to CPUs 0-3 with `OMP_PROC_BIND=close` and `OMP_PLACES=cores`.

| Rows | Updates | Threads | v8 median | v9 median | Time reduction |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 300 | 1 | 0.730 s | 0.716 s | **1.9%** |
| 500,000 | 60 | 1 | 0.729 s | 0.723 s | **0.8%** |
| 1,000,000 | 30 | 1 | 0.757 s | 0.739 s | **2.3%** |
| 1,000,000 | 30 | 4 | 0.389 s | 0.360 s | **7.6%** |

Single-thread batches below 50,000 rows and parallel batches below 1,000,000 rows deliberately retain v8, so no v9 speed-up is claimed for those configurations.

Peak resident memory for the 1,000,000-row, four-thread memory run was 95,616 KiB for v8 and 95,624 KiB for v9, effectively unchanged.

## Numerical validation

All 36 paired release timing runs finished with an identical final scalar checksum. A separate full-state equivalence harness compared every byte of W1, W2 and both momentum arrays after the active v9 path. The 100,000-row one-thread case and 1,000,000-row four-thread case were both byte-identical to v8.

The guarded path uses the same vector exponential operation as v8. The range proof only determines whether the existing `-700` exceptional branch can be omitted safely.

## Progress across versions

V1-v6 historical timings used an earlier AMD EPYC environment, while v7-v9 use the Xeon environment above. The table therefore extends the hardware-normalised ratios rather than mixing absolute seconds from different machines.

| Rows | Normalised v8 vs v1 | Paired v9 vs v8 | Normalised v9 vs v1 |
|---:|---:|---:|---:|
| 13,853 | 9.52x | unchanged | **9.52x** |
| 100,000 | 11.35x | 1.020x | **11.57x** |
| 500,000 | 14.21x | 1.008x | **14.33x** |
| 1,000,000 | 14.78x | 1.023x | **15.12x** |

For four threads, the v9 dispatch is unchanged from v8 below 1,000,000 rows. The one-million-row normalised ratio rises from about 47.1x to **50.95x** versus v1. These are derived progress ratios, not direct v1-versus-v9 measurements.

## Experiments rejected

V9 testing also covered network alignment/aligned weight loads, direct sigmoid dispatch in isolation, explicit OpenMP block ownership in isolation and cross-tile gradient-register carry. Alignment and ownership were inconsistent or below the noise floor. Direct dispatch became useful only as part of the full-tile path. Cross-tile gradient carry was rejected after generated-code inspection showed the current kernel already has substantial ZMM stack spilling, so extending live vector state would increase register pressure.

Third-party vector-math libraries were not benchmarked because oneMKL, SLEEF and AOCL-LibM were not installed in the test environment.

Processed results, raw timings, equivalence evidence, memory measurements and experiment notes are under [`benchmark/`](benchmark/).
