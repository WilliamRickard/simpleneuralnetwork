# Simple Neural Network v13

V13 adds a deliberately narrow BF16 optimisation on top of v12. It keeps FP64 master weights, FP64 momentum, FP32 activations, FP32 sigmoid evaluation, FP32 backpropagation and FP32 gradient accumulation. Only the operands of the first 11 x 16 layer multiply are quantised to BF16.

## Retained design

For each input row, the 11 input features are packed into six BF16 pairs. The 11 x 16 W1 matrix is similarly packed into six vectors of 16 BF16 pairs, with the unused twelfth feature padded by zero.

AVX512_BF16 `VDPBF16PS` then evaluates two feature products per hidden node per instruction and accumulates directly into FP32. The first layer therefore uses six BF16 dot-product steps instead of eleven FP32 FMAs for each row.

Everything after the hidden preactivation remains v12-style FP32:

- adaptive sigmoid;
- W2 forward calculation;
- output delta;
- hidden delta;
- W1 and W2 gradient accumulation.

Per-thread gradients are promoted to FP64 before the inherited momentum update. Master weights and momentum remain FP64, and target confirmation still uses the inherited exact FP64 metric path.

## Dispatch

V13 is intentionally retained only for:

- one training thread;
- at least 50,000 rows;
- x86-64/GCC builds with glibc libmvec;
- runtime AVX512_BF16 support.

The retained single-thread full-tile kernel processes 16 rows per packed W1 load. All other workloads delegate to v12 unchanged.

Four-thread BF16 variants were tested but rejected. Results were too sensitive to host scheduling: depending on sample length and paired-versus-separate medians, the apparent gain ranged from low single digits to low teens. V12 therefore remains the production four-thread path.

## Performance versus v12

Benchmark host:

- AMD EPYC 9V74
- GCC 14.2.0
- Linux x86-64
- AVX-512 and AVX512_BF16
- glibc libmvec

Paired alternating release results:

| Rows | Updates | Threads | v12 median | v13 median | Reduction from medians | Median paired reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 100 | 1 | 0.120 s | 0.107 s | **11.0%** | **11.3%** |
| 500,000 | 40 | 1 | 0.255 s | 0.225 s | **11.8%** | **11.5%** |
| 1,000,000 | 30 | 1 | 0.365 s | 0.329 s | **9.8%** | **10.6%** |

Absolute seconds are host-specific. The paired reductions are the more useful release comparison.

## Numerical drift

V13 is intentionally less precise than v12 because both W1 and first-layer input operands are rounded to BF16 before the hidden preactivation.

Across the release matrix:

- maximum final-weight absolute difference versus v12: `3.42e-9`;
- largest exact-sigmoid RMSE difference: about `1.05e-9`.

Long-run checks:

- 100,000 rows / 1,000 updates / 1 thread: max weight difference about `3.43e-8`; exact RMSE `0.0924110006889` vs `0.0924109897693`;
- 1,000,000 rows / 300 updates / 1 thread: max weight difference about `1.04e-8`; exact RMSE `0.0989500428530` vs `0.0989500390117`.

These are empirical trajectory measurements on the deterministic benchmark, not universal error guarantees.

## Memory

V13 retains v12's FP32 training copy because gradients still use FP32 inputs. It adds six packed 32-bit BF16 feature-pair words per row for the first-layer forward path.

At one million rows:

- v12 FP32 training copy: about 45.8 MiB;
- v13 packed BF16 pair addition: about 22.9 MiB;
- total v13 training-side copy: about 68.7 MiB before allocator overhead.

The BF16 structure is created only when the v13 single-thread path is selected.

## Rejected variants

- BF16 forward plus BF16 paired W1-gradient accumulation: sometimes another 1-3 percentage points faster single-threaded, but inconsistent across larger workloads and substantially less attractive on four threads.
- BF16 four-thread production dispatch: rejected because paired results were not stable enough.
- Reconstructing FP32 gradient inputs from BF16 storage: removed the extra FP32 input copy but erased most of the speed gain.

## Build

Portable:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v13/main.cpp -o simple_nn_v13
```

Accelerated:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    -DSIMPLE_NN_USE_LIBMVEC v13/main.cpp -lm -o simple_nn_v13
```

No global `-mavx512bf16` flag is required for production. The BF16 kernel uses a function target attribute plus a runtime capability check, so unsupported CPUs remain on v12.

## What remains

V13 shows that BF16 can still buy roughly another 10-12% on the single-thread hot path, but the return is much smaller than the FP32 transition in v12. Larger future gains are now more likely to come from changing the training algorithm or dataflow than from another precision step.
