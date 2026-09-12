# Simple Neural Network v3

V3 is a performance-focused continuation of the original 2017 program and the v2 rewrite. The network mathematics are unchanged:

```text
11 inputs -> 16 sigmoid hidden nodes -> 1 sigmoid output
```

As with the earlier versions, the complete implementation is kept in one `main.cpp` so the full program can be read from top to bottom.

## What changed from v2

The main v3 change is a fused training pass. V2 performs forward propagation for the whole batch, stores every hidden activation and prediction, then walks the whole batch again for backpropagation. V3 completes forward propagation, diagnostics and backpropagation for one observation before moving to the next.

That means the training path no longer needs a `rows x 16` hidden-activation matrix or a full prediction vector. Each observation uses only two small 16-value scratch arrays for the hidden activations and hidden-layer delta.

The fixed 11-input and 16-hidden-node dimensions are also compile-time constants. This gives the compiler more information about the small inner loops while keeping the code straightforward.

The sigmoid path has been simplified for the normal numeric range:

```text
1 / (1 + exp(-x))
```

For extremely negative values v3 switches to the algebraically equivalent stable form so `exp(-x)` cannot overflow.

Inference no longer allocates a hidden matrix for every observation. Predictions are calculated one row at a time and written directly to `ybar.txt`.

Finally, v3 can optionally parallelise training observations with OpenMP. Each thread owns a private gradient accumulator, so there are no locks or atomics in the hot row loop. The thread-local gradients are combined after the pass in a fixed thread order.

## Building

Portable single-threaded build:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v3/main.cpp -o simple_nn_v3
```

Optional OpenMP build:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp v3/main.cpp -o simple_nn_v3
```

The source defaults to `trainingThreads = 1`. Increase that value near the top of `main()` when using the OpenMP build.

`-march=native`, `-fno-math-errno`, `-ffast-math` and profile-guided optimisation were also tested while developing v3. None produced a sufficiently consistent improvement in the benchmark environment to become part of the recommended build.

## Benchmark results

The benchmark uses the same deterministic synthetic 11-input data and starting weights as the v2 benchmark. Dataset generation and file I/O are outside the timed training section. Both versions perform the same fixed number of full-batch gradient updates.

| Rows | Updates | v2 median | v3 single-thread | Single-thread speed-up | v3 4-thread | 4-thread speed-up |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 13,853 | 500 | 1.270 s | 1.091 s | 1.16x | 0.331 s | 3.84x |
| 100,000 | 100 | 1.714 s | 1.490 s | 1.15x | 0.765 s | 2.24x |
| 500,000 | 20 | 1.766 s | 1.515 s | 1.17x | 0.854 s | 2.07x |
| 1,000,000 | 10 | 1.804 s | 1.525 s | 1.18x | 0.773 s | 2.33x |

On the 1,000,000-row case, peak resident memory fell from about 223 MiB for v2 to about 93 MiB for v3, a further reduction of roughly 58%.

The serial v3 benchmark produced the same final weight checksum as v2 in every measured case. The four-thread version also produced the same reported checksum. Its different floating-point reduction order changes aggregate diagnostics only at round-off level, with the largest observed cost difference about `1.4e-10`.

The benchmark machine exposed five AMD EPYC 9V74 CPU cores. Four worker threads gave the best stable result in the tests, so the table above uses four threads rather than assuming all machines should use the same setting.

Full results and raw timings are under [`benchmark/`](benchmark/).

## Why single-thread performance improved less than memory

V2 had already removed the largest transpose and temporary-matrix costs. After the v3 fusion, the remaining hot path is dominated by sigmoid evaluation. Each observation still requires 16 hidden sigmoids and one output sigmoid, so a million-row update performs 17 million calls to `exp()`.

That is why v3 gives a further 15-18% single-thread speed-up rather than another multi-fold gain. More aggressive sigmoid approximations or reduced precision could go faster, but they would change the numerical characteristics of the model and are deliberately not part of this version.
