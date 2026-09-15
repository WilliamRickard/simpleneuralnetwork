# v28 benchmark

The production exactness harness includes `v28/main.cpp` directly with its executable `main()` disabled, so the benchmark exercises the exact `evaluateSliceV28` implementation committed for release rather than a copied kernel.

Build the accelerated benchmark with the same strict C++11 warning policy used for prior releases:

```bash
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -Werror \
    -fopenmp -DSIMPLE_NN_USE_LIBMVEC \
    v28/benchmark/evaluator_exactness_benchmark.cpp \
    -lmvec -o /tmp/v28-evaluator-exact
```

Then run:

```bash
/tmp/v28-evaluator-exact
```

On an AVX-512 host the expected result is:

```text
PASS: 425/425 v17-v28 production-kernel cases are exact
```

The 425 cases are 17 row counts × 5 deterministic seeds × 5 worker partitions. Each partition compares squared error, percentage-error sum, maximum percentage error and all 192 gradient doubles. The harness exits non-zero on the first mismatch.

On a host without the accelerated GCC/glibc AVX-512 path, the benchmark exits successfully with an explicit `SKIP` message.
