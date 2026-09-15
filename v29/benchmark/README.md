# v29 benchmark and release gates

V29 has two in-repository production-source benchmarks. Both include `v29/main.cpp` directly, so they exercise the committed evaluator and optimiser wrappers rather than copied kernels.

## 1. Evaluator exactness

Compile with the same strict accelerated C++11 policy used for recent releases:

```bash
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -Werror \
    -fopenmp -DSIMPLE_NN_USE_LIBMVEC \
    v29/benchmark/evaluator_exactness_benchmark.cpp \
    -lmvec -o /tmp/v29-evaluator-exact
```

Run:

```bash
/tmp/v29-evaluator-exact
```

On a supported AVX-512 GCC/glibc host the release result was:

```text
PASS: 425/425 configurations and 1275 slice comparisons are exact
```

This fixed matrix covers five deterministic seeds, 17 awkward row counts through 100,000, and one through five worker partitions. It compares squared error, percentage-error sum, maximum percentage error, and all 192 gradient doubles between `evaluateSliceV28` and `evaluateSliceV29`.

The release investigation also ran a randomized differential stress with 4,000/4,000 exact comparisons across start offsets, feature scales, weight scales, zero targets and large slices. Those exploratory cases are documented in `../README.md` but are not duplicated in this compact checked-in harness.

## 2. Full-training matrix

The final release matrix was run with:

```bash
python3 v29/benchmark/run_full_training.py \
    --pairs 7 --threads 2 --cooldown-ms 200 \
    --output v29/benchmark/full_training_summary.csv
```

The runner strictly compiles `full_training_benchmark.cpp` with:

```text
-std=c++11 -O3 -Wall -Wextra -Wpedantic -Werror
-fopenmp -DSIMPLE_NN_USE_LIBMVEC -lmvec
```

It defaults `OMP_PROC_BIND=true`, `OMP_PLACES=cores` and `OMP_DYNAMIC=false` unless the caller has already supplied those variables.

It tests both deep targets, `0.001%` and `0.0005%`, at 5k, 10k, 20k, 50k and 100k rows.

Before any timing is accepted, each case runs `trainRangeV28` and `trainRangeV29` from identical initial weights and requires exact equality of:

- accepted update count;
- target-reached status;
- final cost, mean percentage error and maximum percentage error;
- all 192 final parameter doubles.

The synthetic teacher targets are generated with the same AVX-512/libmvec production forward arithmetic, giving the benchmark an exactly representable optimum. A case fails if it reaches the target with zero updates, fails to reach the target, or changes the v28 trajectory.

Timing pairs alternate execution order. A 200 ms idle period is inserted before and between timed runs, outside the measured interval. This avoids the order-dependent frequency and throttling artefacts observed on quota-constrained hosts during v28/v29 research.

## Release result

All ten production cases reached their targets and preserved exact v28-v29 trajectories. V29 won all 70 timed pairs.

| Rows | Target | Updates | Median | p10 | p90 | Wins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 0.001% | 159 | 1.172x | 1.165x | 1.193x | 7/7 |
| 5,000 | 0.0005% | 351 | 1.185x | 1.183x | 1.193x | 7/7 |
| 10,000 | 0.001% | 148 | 1.155x | 1.142x | 1.186x | 7/7 |
| 10,000 | 0.0005% | 352 | 1.161x | 1.157x | 1.179x | 7/7 |
| 20,000 | 0.001% | 147 | 1.164x | 1.157x | 1.183x | 7/7 |
| 20,000 | 0.0005% | 350 | 1.188x | 1.169x | 1.207x | 7/7 |
| 50,000 | 0.001% | 133 | 1.187x | 1.179x | 1.191x | 7/7 |
| 50,000 | 0.0005% | 303 | 1.302x | 1.300x | 1.319x | 7/7 |
| 100,000 | 0.001% | 132 | 1.262x | 1.258x | 1.266x | 7/7 |
| 100,000 | 0.0005% | 305 | 1.322x | 1.305x | 1.334x | 7/7 |

The machine-readable results are retained in `full_training_summary.csv`.

## Committed-source identity and compatibility

The release investigation first validated the no-main include-chain correction in a workspace patch, then committed the same 12-file structural change. A later committed-source build produced byte-identical benchmark executables to the AVX-512-tested build:

```text
93064024d0a63b28a9044583edebdde5ddf8cd53f3ab3ad9a11694f044597e35  v29-evaluator-exact
c1c283a327b932f82030a0c974539f7fd493210ab6b9b03899eb24c2ec601daa  v29-full-training
```

Historical v18-v26 portable and accelerated standalone compatibility builds passed. V27, v28 and v29 passed strict portable and accelerated C++11 builds with `-Wall -Wextra -Wpedantic -Werror`.

V18's older portable source still emits longstanding unused-function warnings if every warning is promoted to an error, so historical v18-v26 compatibility is deliberately a build-success gate rather than a retroactive warning-cleanliness requirement.
