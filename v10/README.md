# Simple Neural Network v10

V10 is the final strict bit-identical optimisation round over v9. It keeps the same 11 -> 16 -> 1 sigmoid network, double-precision arithmetic, libmvec exponential, gradient equations, reduction order and momentum update.

The retained change is deliberately narrow: on supported AVX-512/libmvec builds, **exactly four training threads with a batch of at least 1,000,000 rows** use an eight-observation forward kernel. Every other configuration delegates to v9 unchanged.

## What changed

V9 evaluates four observations at a time in the 11 x 16 forward multiply. For each input feature it loads the two 8-wide W1 vectors and broadcasts four input scalars.

V10 evaluates eight observations at a time. The same two W1 vectors are therefore reused across eight observations before the next feature is loaded. Within every observation the sequence of FMAs remains `k = 0..10`, exactly as in v9. Only independent observations are interleaved differently.

Nothing else in the retained path changes:

- the guarded v9 sigmoid path is reused;
- output-dot reduction order is unchanged;
- output sigmoid is unchanged;
- `deltaThree` and hidden-delta equations are unchanged;
- each W1 and W2 gradient receives rows in the same order and with the same tile grouping;
- the per-thread percentage-error cutoff is unchanged;
- thread-local gradient combination order is unchanged;
- momentum and weight update order is unchanged.

## Dispatch

V10 uses the new kernel only when all of these hold:

- `trainingThreads == 4`;
- batch size is at least 1,000,000 rows;
- the v9 AVX-512/libmvec capability check passes.

Otherwise `trainRangeV10` immediately delegates to v9.

This narrow dispatch is intentional. Screening found the eight-row forward kernel slower for single-thread execution, about 1.3% slower at two threads and about 0.7% slower at five threads. Three threads showed only a small short-run signal, so v9 remains the conservative default there too.

## Release benchmark

Environment is the same machine used for v8/v9 profiling:

- Intel Xeon Platinum 8573C
- GCC 14.2.0
- Linux x86-64
- glibc `libmvec`
- four threads pinned to CPUs 0-3
- `OMP_PROC_BIND=close`
- `OMP_PLACES=cores`

The retained path was tested with 1,000,000 rows and 100 updates, five alternating v9/v10 pairs.

| Metric | v9 | v10 | Improvement |
|---|---:|---:|---:|
| Median elapsed time | 2.489 s | 2.378 s | **4.44% less time** |
| Median paired comparison | - | - | **2.62% less time** |

The paired metric is the more conservative summary because it compares v9 and v10 within each alternating pair before taking the median.

## Exactness

All five long release pairs finished with **byte-identical complete network state**, comparing W1, W2 and both momentum arrays. The final checksum in every pair was `0.25546753619194329` for both versions.

The deterministic 1,000,000-row / 30-update checksum from the v10 development harness also matches the committed v9 benchmark checksum exactly: `0.2610809529780989`.

## Rejected v10 experiments

V10 also tested gradient-register splitting, isolated no-inline gradient helpers, GCC PGO, `-march=native`, explicit `-march=emeraldrapids`, and broader thread-count use of the eight-row kernel. None met the retention bar. Details are in [`benchmark/experiments.txt`](benchmark/experiments.txt).

The important result is that reducing gradient-helper register pressure was mechanically possible without changing bits, but the extra passes over each tile cost more than the spill reduction saved. PGO was inconsistent. CPU-specific `-march` builds were unsafe on the virtualised runner and were rejected rather than shipped.

## Build

Portable:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v10/main.cpp -o simple_nn_v10
```

OpenMP + glibc vector-exp:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    -DSIMPLE_NN_USE_LIBMVEC v10/main.cpp -lm -o simple_nn_v10
```

V10 layers over the frozen v9 source in the same way v9 layers over v8.

## Exact-performance frontier

On this CPU and benchmark, v10 suggests that the strict byte-identical source kernel is now close to saturated. Further substantial gains are more likely to require relaxing the equivalence target, for example allowing a tightly bounded ULP difference in vector exponential or reduction order, rather than further rearranging the exact arithmetic.
