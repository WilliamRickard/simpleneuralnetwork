# V4 benchmark

This directory records the measurements used to compare v4 with the committed v3 implementation.

## What changed

V4 targets the main remaining v3 bottleneck: scalar sigmoid evaluation.

The benchmarked v4 kernel:

- processes observations in blocks of 64;
- evaluates the block's hidden and output sigmoids through glibc `libmvec` when compiled with `SIMPLE_NN_USE_LIBMVEC` on supported x86/glibc systems;
- selects AVX-512, AVX2 or the scalar fallback at runtime;
- keeps one OpenMP worker team alive across repeated gradient descents;
- reuses the hidden block as `deltaTwo` storage during backpropagation.

A block-only prototype did not produce a useful speed-up. The material improvement came from vectorising `exp()` / sigmoid. The persistent OpenMP team was retained because it avoids repeated parallel-region setup in the combined fast path.

## Method

The workloads match the deterministic synthetic benchmark used for v3:

| Rows | Updates |
|---:|---:|
| 13,853 | 500 |
| 100,000 | 100 |
| 500,000 | 20 |
| 1,000,000 | 10 |

The v3 medians are the values already committed in `v3/benchmark/benchmark_summary.csv`. V4 was measured in the same execution environment and with the same model dimensions, initial weights, learning rate and momentum. V4 medians are from three repeated runs of each workload.

Compiler settings for the accelerated benchmark:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    -DSIMPLE_NN_USE_LIBMVEC ... -lm
```

The benchmark harness uses the same runtime-dispatched AVX-512 / AVX2 vector-exp approach as `v4/main.cpp`.

## Results

| Rows | Updates | v3 single | v4 single | Single speed-up | v3 4-thread | v4 4-thread | 4-thread speed-up |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13,853 | 500 | 1.091 s | 0.581 s | 1.88x | 0.331 s | 0.260 s | 1.27x |
| 100,000 | 100 | 1.490 s | 0.876 s | 1.70x | 0.765 s | 0.312 s | 2.45x |
| 500,000 | 20 | 1.515 s | 0.902 s | 1.68x | 0.854 s | 0.360 s | 2.37x |
| 1,000,000 | 10 | 1.525 s | 0.862 s | 1.77x | 0.773 s | 0.345 s | 2.24x |

The exact processed values are in `benchmark_summary.csv`.

## Memory

The one-million-row v4 case used about 93 MiB peak RSS in both single-thread and four-thread runs, effectively unchanged from v3. The 64-row scratch block is small enough that the vectorisation gain does not materially increase working memory.

See `memory.txt` for the measured values.

## Numerical validation

The glibc AVX-512 vector `exp()` implementation was compared with scalar `std::exp()` over 800,000 random double values uniformly sampled from [-50, 50]. The maximum relative difference was about `4.42e-16` and the maximum difference was 3 ULP.

A separate production-path test used identical text input and starting weights and performed one update with v3 and accelerated v4. The resulting `wone.txt`, `wtwo.txt` and `ybar.txt` files were byte-identical at their stored precision.

The controlled benchmark also finished on the same printed weight checksum as v3 for the corresponding workloads. Parallel reductions can differ at round-off level because floating-point additions are grouped by worker.

See `vector_exp_accuracy.txt` for the recorded accuracy check.

## Portability

`libmvec` is an optional accelerated backend rather than a requirement of v4. The ordinary C++11 build remains available and uses the scalar sigmoid. On a non-glibc or non-x86 machine, the vector backend is not compiled.

A future cross-platform extension could add SLEEF or another vector-math backend behind the same `sigmoidVector()` interface.
