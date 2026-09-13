# Simple Neural Network v11

V11 is the first deliberately non-bit-identical optimisation round. It keeps the v10 network architecture, double-precision weights, full-batch momentum update, backpropagation equations and gradient-reduction structure, but replaces the expensive vector exponential inside the training sigmoid on selected hot paths.

Inference and target confirmation continue to use the existing exact sigmoid path.

## Retained change

The final v11 training sigmoid is deliberately simpler than the earlier multi-region prototype.

For each AVX-512 vector:

- if every lane satisfies `|x| <= 1`, evaluate
  `0.5 + x * (c1 + x^2 * (c3 + x^2 * c5))`;
- otherwise fall back to the existing checked libmvec sigmoid for that vector.

The retained coefficients are:

- `c1 = 0.24998101634651657`
- `c3 = -0.020677835421401624`
- `c5 = 0.0017580292406143272`

A dense 1,000,001-point validation grid on `[-1, 1]` gives maximum absolute sigmoid error about `2.69e-6`. Outside `[-1,1]` the existing sigmoid is used, so this is also the global approximation-error bound apart from ordinary floating-point evaluation error.

The previous broader piecewise-polynomial design was rejected for production because the measured training activations are overwhelmingly central and exact fallback is both simpler and safer.

## Dispatch

V11 uses the adaptive sigmoid only on production paths benchmarked directly:

- one training thread with at least 50,000 rows: four-observation forward kernel plus v11 sigmoid;
- exactly four training threads with at least 1,000,000 rows: eight-observation v10 forward kernel plus v11 sigmoid.

All other configurations delegate to v10 unchanged.

The existing AVX-512/libmvec capability gate is retained. Portable builds therefore continue to use the exact historical path.

## Performance

The final release-development benchmark was run on the current execution host:

- AMD EPYC 9V74
- GCC 14.2.0
- Linux x86-64
- AVX-512 available
- glibc libmvec
- paired alternating v10-style exact-sigmoid and v11 adaptive-sigmoid runs

Absolute seconds are specific to this runner. The paired reductions are the useful comparison.

| Rows | Updates | Threads | v10-style median | v11 median | Reduction from medians | Median paired reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 100 | 1 | 0.331 s | 0.207 s | **37.7%** | **37.8%** |
| 500,000 | 40 | 1 | 0.659 s | 0.417 s | **36.8%** | **36.6%** |
| 1,000,000 | 30 | 1 | 1.014 s | 0.641 s | **36.8%** | **36.7%** |
| 1,000,000 | 60 | 4 | 0.640 s | 0.394 s | **38.5%** | **30.9%** |

The four-thread cell is noisier, so the paired-median reduction is the more conservative headline there.

## Numerical drift

V11 is intentionally not byte-identical to v10. The measured drift is nevertheless very small on the deterministic benchmark when both networks are evaluated afterwards with the exact scalar sigmoid.

Release cells:

- maximum final-weight absolute difference across the release matrix: `5.12e-9`;
- largest exact-sigmoid RMSE difference across the release matrix: below `1.0e-8`.

Longer stability checks:

- 100,000 rows / 1,000 updates / 1 thread: max weight difference `6.02e-8`; exact RMSE `0.09241115536` vs `0.09241103897`;
- 1,000,000 rows / 300 updates / 4 threads: max weight difference `1.62e-8`; exact RMSE `0.09895008445` vs `0.09895005305`.

These are empirical trajectory comparisons, not a guarantee for arbitrary datasets.

## Stopping semantics

Training gradients and the lightweight percentage metric use the approximate sigmoid on the v11 path. If the approximate metric indicates that the target may have been reached, v11 recomputes the metrics through the existing exact inference path before accepting the stopping condition. Thus the approximation cannot by itself produce a false-positive target success.

## Rejected architecture change

A row-oriented structure-of-arrays blocked implementation was tested. It reduces gradient-store traffic and permits reassociation across rows, but this network is only `11 x 16`. The extra scalar-weight broadcasts and horizontal reductions outweighed the storage savings. Even with the fast sigmoid it remained slower than the existing hidden-node SIMD layout, so it is not retained.

## Build

Portable:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v11/main.cpp -o simple_nn_v11
```

OpenMP + glibc vector-exp:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    -DSIMPLE_NN_USE_LIBMVEC v11/main.cpp -lm -o simple_nn_v11
```

## What remains

V11 deliberately stops before changing precision or the optimiser. The next larger performance steps are FP32/mixed precision, fast-math/reassociation experiments, and changing the optimisation algorithm itself. Those alter the numerical/training contract more substantially and should be measured as separate versions.
