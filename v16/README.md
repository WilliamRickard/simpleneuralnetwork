# v16 - AVX-512 FP64 L-BFGS evaluator

V16 keeps the v15 optimiser and four-worker layout, but vectorises the exact full-batch objective/gradient evaluator across the 16 hidden nodes.

## Retained design

- same 11 -> 16 -> 1 sigmoid network
- same full-batch L-BFGS history and Armijo line search as v14/v15
- same four-worker cap as v15
- FP64 weights, activations, deltas and gradients
- hidden layer mapped to two 8-double AVX-512 vectors
- eight observations share each W1 vector load
- 11 low/high W1 gradient vector pairs plus the W2 gradients stay in registers across each worker slice
- FP64 libmvec vector `exp` is used for the hidden sigmoid
- the inherited scalar FP64 metric path still performs the final stopping confirmation
- unsupported builds/CPUs fall back to v15

The internal SIMD evaluator is FP64 but is not intended to be bit-identical to the scalar v15 evaluator because libmvec and the reassociated vector reductions can differ in the last bits.

## Time to target versus v15

Seven alternating paired runs on the same AMD EPYC 9V74 host:

| Rows | v15 median | v16 median | Reduction from medians | Median paired reduction |
|---:|---:|---:|---:|---:|
| 100,000 | 0.03245 s | 0.00862 s | 73.4% | 73.1% |
| 500,000 | 0.19039 s | 0.07447 s | 60.9% | 62.6% |
| 1,000,000 | 0.41834 s | 0.14781 s | 64.7% | 64.7% |

Both v15 and v16 reached the 3.9% target in exactly three evaluations in every retained run. Final percentage errors agreed to about machine precision in the benchmark.

## Kernel tuning

The first SIMD kernel processed one row per W1 load. Grouping four rows improved it materially. Eight-row grouping was the best overall release choice. Sixteen-row grouping was competitive at 1m rows but less consistent at smaller sizes and placed more pressure on the Zen 4 register file.

## Build

Portable fallback:

```bash
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v16/main.cpp -o simple_nn_v16
```

Accelerated path:

```bash
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp -DSIMPLE_NN_USE_LIBMVEC v16/main.cpp -lm -o simple_nn_v16
```

The accelerated path runtime-checks AVX-512 support and otherwise delegates to v15.
