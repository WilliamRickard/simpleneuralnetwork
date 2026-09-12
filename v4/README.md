# Simple Neural Network v4

V4 keeps the original 11 -> 16 -> 1 sigmoid network and the same training equations, while targeting the remaining hot path in v3: repeated scalar sigmoid/`exp()` evaluation.

## Main changes

- Processes training rows in blocks of 64 so hidden activations are contiguous for vector maths.
- Adds an optional glibc `libmvec` backend with runtime AVX2 / AVX-512 dispatch.
- Keeps the scalar sigmoid as the portable fallback.
- Keeps one OpenMP worker team alive across gradient descents instead of creating a new parallel region for every update.
- Reuses the block hidden buffer for `deltaTwo`, so peak memory remains close to v3.
- Preserves the one-file implementation style in `main.cpp`.

## Builds

Portable C++11 build:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v4/main.cpp -o simple_nn_v4
```

OpenMP build:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp v4/main.cpp -o simple_nn_v4
```

Accelerated glibc/x86 build used for the v4 benchmark:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    -DSIMPLE_NN_USE_LIBMVEC v4/main.cpp -lm -o simple_nn_v4
```

The accelerated build does not require compiling the whole program for AVX2 or AVX-512. The vector sigmoid functions use targeted code paths and select AVX-512, AVX2 or the scalar fallback at runtime.

## Performance

The benchmark uses the same deterministic synthetic workloads as v3.

| Rows | Updates | v3 single | v4 single | Speed-up | v3 4-thread | v4 4-thread | Speed-up |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13,853 | 500 | 1.091 s | 0.581 s | 1.88x | 0.331 s | 0.260 s | 1.27x |
| 100,000 | 100 | 1.490 s | 0.876 s | 1.70x | 0.765 s | 0.312 s | 2.45x |
| 500,000 | 20 | 1.515 s | 0.902 s | 1.68x | 0.854 s | 0.360 s | 2.37x |
| 1,000,000 | 10 | 1.525 s | 0.862 s | 1.77x | 0.773 s | 0.345 s | 2.24x |

### Progress across versions

The v1 and v2 values below are the original paired baseline medians stored under `v2/benchmark/`. V3 and v4 were measured in later benchmark rounds using the same workload sizes and model. This table is intended to show cumulative progress through the versions, while the table above remains the stricter paired v3-versus-v4 comparison.

| Rows | Updates | v1 baseline | v2 baseline | v3 single | v4 single | v4 single vs v1 | v4 4-thread | v4 4-thread vs v1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13,853 | 500 | 2.289 s | 1.172 s | 1.091 s | 0.581 s | 3.94x | 0.260 s | 8.81x |
| 100,000 | 100 | 3.737 s | 1.682 s | 1.490 s | 0.876 s | 4.27x | 0.312 s | 11.96x |
| 500,000 | 20 | 4.838 s | 1.782 s | 1.515 s | 0.902 s | 5.36x | 0.360 s | 13.42x |
| 1,000,000 | 10 | 4.975 s | 1.830 s | 1.525 s | 0.862 s | 5.77x | 0.345 s | 14.41x |

V2 originally delivered about a 1.95x to 2.72x speed-up over v1. By v4, the single-thread path is about 3.94x to 5.77x faster than v1 across these workloads, while the four-thread path reaches about 8.81x to 14.41x.

Peak RSS for the one-million-row case remained about 93 MiB, essentially unchanged from v3.

## Numerical validation

The glibc vector `exp()` backend was compared against scalar `std::exp()` over 800,000 random double values in [-50, 50]. The maximum relative difference was about `4.42e-16`, with a maximum difference of 3 ULP.

A production-path one-update comparison using identical text input and weights produced byte-identical `wone.txt`, `wtwo.txt` and `ybar.txt` between v3 and accelerated v4.

Detailed results and raw validation notes are in `v4/benchmark/`.
