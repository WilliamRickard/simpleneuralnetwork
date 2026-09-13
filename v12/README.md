# Simple Neural Network v12

V12 is the first mixed-precision optimisation round. It keeps the v11 network architecture, full-batch gradient descent, FP64 master weights, FP64 momentum state, stopping target and exact inference path, but evaluates the hot training path in FP32.

## Retained design

On retained v12 training paths:

- the input matrix and targets are copied once to FP32;
- each pass converts the current 192 FP64 master weights to FP32;
- forward propagation, adaptive sigmoid, backpropagation and per-thread gradient accumulation run in FP32;
- each 16-row tile keeps the 11 W1 gradient vectors and the W2 gradient vector in AVX-512 registers until the end of the tile;
- per-thread FP32 gradients are promoted to FP64 before the historical momentum update;
- master weights and momentum remain FP64.

This uses all 16 hidden nodes in one 512-bit FP32 vector, compared with two eight-double vectors in v11. It also halves the hot dataset and gradient bandwidth.

The v11 adaptive sigmoid contract is retained. A vector entirely inside `[-1,1]` uses the short degree-5 odd polynomial; otherwise v12 calls the 16-wide glibc `expf` vector function and computes the exact FP32 logistic expression.

## Dispatch

V12 is used only on paths benchmarked directly:

- one training thread with at least 50,000 rows;
- exactly four training threads with at least 1,000,000 rows.

All other paths delegate to v11 unchanged. Portable builds without the AVX-512/libmvec capability gate therefore remain on the historical FP64 path.

## Performance

Benchmark host:

- AMD EPYC 9V74
- GCC 14.2.0
- Linux x86-64
- AVX-512
- glibc libmvec

The benchmark compares a production-shaped v11 FP64/adaptive-sigmoid kernel with the retained v12 FP64-master/FP32-compute kernel on the same host.

| Rows | Updates | Threads | v11 median | v12 median | Reduction from medians | Median paired reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 100 | 1 | 0.220 s | 0.123 s | **44.3%** | **44.2%** |
| 500,000 | 40 | 1 | 0.457 s | 0.243 s | **46.8%** | **46.0%** |
| 1,000,000 | 30 | 1 | 0.651 s | 0.364 s | **44.1%** | **44.0%** |
| 1,000,000 | 60 | 4 | 0.395 s | 0.209 s | **47.2%** | **50.0%** |

Absolute seconds are host-specific. The paired reductions are the more useful release comparison.

## Numerical drift

V12 is intentionally not numerically identical to v11 because activations and gradient accumulation use FP32. The FP64 master-weight design keeps the observed trajectory extremely close on the deterministic benchmark.

Release-shaped cells:

- maximum final-weight absolute difference: `7.68e-10`;
- maximum exact-sigmoid RMSE difference: about `1.19e-11`.

Long-run checks:

- 100,000 rows / 1,000 updates / 1 thread: max weight difference `1.33e-10`; exact RMSE `0.0924110006907` vs `0.0924110006889`;
- 1,000,000 rows / 300 updates / 4 threads: max weight difference `4.50e-11`; exact RMSE `0.0989500428421` vs `0.0989500428340`.

These are empirical trajectory measurements, not universal error bounds for arbitrary datasets.

## Why full FP32 was rejected

A full-FP32 variant, including FP32 master weights and momentum, was also benchmarked. It was essentially the same speed as the FP64-master mixed path, but its weight drift was about three orders of magnitude larger in the release cells and reached around `1e-6` to `3e-6` in the long-run checks. V12 therefore keeps the master optimiser state in double precision.

## Stopping semantics

Training gradients and the lightweight percentage-error metric use the FP32 training path. If the approximate metric indicates the target may have been reached, v12 recomputes the metrics through the inherited exact FP64 inference path before accepting success. This preserves v11's protection against false-positive target crossings.

## Memory

The retained path keeps the original FP64 dataset for exact metrics/inference and adds an FP32 training copy. At one million rows with 11 inputs plus one target this adds about 48 MB decimal, or 45.8 MiB, before allocator overhead. Small and unsupported workloads do not build the FP32 copy.

## Build

Portable:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v12/main.cpp -o simple_nn_v12
```

Accelerated:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    -DSIMPLE_NN_USE_LIBMVEC v12/main.cpp -lm -o simple_nn_v12
```

## What remains

The next numerical-contract step would be BF16 for matrix operands with FP32 accumulation, which this Zen 4 host supports through AVX512_BF16. That should be treated separately because it materially reduces mantissa precision. A separate algorithmic branch could instead investigate mini-batches or L-BFGS, where time-to-target rather than time-per-update becomes the primary metric.
