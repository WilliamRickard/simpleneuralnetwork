# Simple Neural Network v11

V11 is the first deliberately non-bit-identical optimisation round. It keeps the v10 network architecture, double-precision weights, full-batch momentum update and backpropagation equations, but replaces the expensive vector exponential inside the training sigmoid with a bounded AVX-512 polynomial approximation on selected hot paths.

Inference, final prediction files and target confirmation continue to use the existing exact sigmoid path.

## Retained change

The v11 training sigmoid uses symmetry, `sigmoid(-x) = 1 - sigmoid(x)`, and approximates only `|x|`:

- `|x| <= 1`: degree-4 polynomial
- `1 < |x| <= 4`: degree-8 polynomial
- `4 < |x| <= 12`: degree-8 polynomial
- `|x| > 12`: saturate to 0 or 1

A dense validation grid on `[-20, 20]` gives a maximum absolute sigmoid error below `6.57e-6`. The approximation remains inside `[0, 1]` on that grid.

On the deterministic million-row benchmark after 100 exact-sigmoid updates, the largest hidden preactivation was about `0.661` and the largest output preactivation about `0.102`, so the short degree-4 common path handles the measured hot workload.

## Dispatch

V11 uses the approximate sigmoid only on paths that were benchmarked directly:

- one training thread with at least 50,000 rows: v9-style four-observation forward kernel plus v11 sigmoid
- exactly four training threads with at least 1,000,000 rows: v10 eight-observation forward kernel plus v11 sigmoid

All other configurations delegate to v10 unchanged.

The existing AVX-512/libmvec capability gate is retained. This makes v11 an opt-in accelerated path on the same build/environment used by v9 and v10 rather than broadening the platform contract at the same time as changing numerical behaviour.

## Performance

The v11 release-development benchmark is a paired production-shaped isolation harness. It mirrors the four-row single-thread and eight-row four-thread forward structures and changes the sigmoid implementation while keeping the same synthetic dataset, gradient equations and momentum update. Absolute seconds should not be compared directly with the historical v10 README because the harness bookkeeping is not byte-for-byte the same. The paired percentage reductions are the useful metric.

| Rows | Updates | Threads | v10-style median | v11 median | Time reduction | Median paired reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 300 | 1 | 1.231 s | 0.692 s | **43.8%** | **44.0%** |
| 500,000 | 60 | 1 | 1.040 s | 0.693 s | **33.4%** | **33.6%** |
| 1,000,000 | 30 | 1 | 1.037 s | 0.701 s | **32.4%** | **31.9%** |
| 1,000,000 | 100 | 4 | 1.086 s | 0.804 s | **26.0%** | **26.0%** |

The gain is largest in the single-thread paths because vector exponential represented a larger share of total runtime there. The four-thread path still improves materially after v10's forward-kernel optimisation.

## Numerical drift

V11 is intentionally not byte-identical to v10. The measured drift is nevertheless very small on the deterministic benchmark when both trained networks are evaluated afterwards with the exact scalar sigmoid.

Examples:

- 100,000 rows / 300 updates / 1 thread: maximum final weight difference `2.39e-8`
- 1,000,000 rows / 30 updates / 1 thread: maximum final weight difference `1.98e-9`
- 1,000,000 rows / 100 updates / 4 threads: maximum final weight difference `7.32e-9`
- 100,000 rows / 1,000 updates / 1 thread: maximum final weight difference `8.96e-8`
- 1,000,000 rows / 300 updates / 4 threads: maximum final weight difference `2.39e-8`

At 100,000 rows / 1,000 updates, exact-sigmoid RMSE was `0.0924111554` for the v10-style baseline and `0.0924109855` for v11. At 1,000,000 rows / 300 updates / four threads it was `0.0989500844` versus `0.0989500393`.

These are empirical results, not a guarantee for arbitrary datasets. The approximation error bound applies to the sigmoid function itself, while optimisation trajectories can amplify small numerical differences over long training runs.

## Stopping semantics

V11 does not accept an approximate false positive at the percentage-error target. When an approximate training pass indicates that the target may have been reached, v11 recalculates metrics through the existing exact inference path before accepting the stopping condition. The final maximum-descent report is also recalculated exactly.

## Rejected architecture change

A row-oriented structure-of-arrays blocked implementation was tested at 64-row and 128-row blocks. It reduces gradient-store traffic and permits reassociation across rows, but this network is only `11 x 16`. The extra scalar-weight broadcasts and horizontal reductions outweighed the storage savings. Even after replacing libmvec with the polynomial sigmoid, the blocked kernel remained slower than the existing hidden-node SIMD layout, so it is not retained.

## Build

Portable build, which delegates unsupported paths to v10:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v11/main.cpp -o simple_nn_v11
```

Accelerated build:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    -DSIMPLE_NN_USE_LIBMVEC v11/main.cpp -lm -o simple_nn_v11
```

The production source was interface-compiled in both modes before being committed. The exact repository-shaped build is also checked separately before merge.

## What remains

V11 deliberately stops before changing precision or the optimiser. The next larger performance steps are likely to be FP32/mixed precision, compiler fast-math/reassociation experiments, or changing the optimisation algorithm itself. Those alter the numerical/training contract more substantially and should be measured as separate versions.
