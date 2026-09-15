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

On a supported AVX-512 GCC/glibc host the expected result is:

```text
PASS: 425/425 configurations and 1275 slice comparisons are exact
```

This fixed matrix covers five deterministic seeds, 17 awkward row counts through 100,000, and one through five worker partitions. It compares squared error, percentage-error sum, maximum percentage error, and all 192 gradient doubles between `evaluateSliceV28` and `evaluateSliceV29`.

The release investigation also ran an external randomized differential stress with 4,000/4,000 exact comparisons across start offsets, feature scales, weight scales, zero targets and large slices. Those exploratory cases are documented in `../README.md` but are not duplicated in this compact checked-in harness.

## 2. Full-training matrix

Run the complete v28-v29 production wrapper matrix with:

```bash
python3 v29/benchmark/run_full_training.py --pairs 7 --threads 4
```

The runner strictly compiles `full_training_benchmark.cpp` with:

```text
-std=c++11 -O3 -Wall -Wextra -Wpedantic -Werror
-fopenmp -DSIMPLE_NN_USE_LIBMVEC -lmvec
```

It then tests both deep targets, `0.001%` and `0.0005%`, at 5k, 10k, 20k, 50k and 100k rows.

Before any timing is accepted, each case runs `trainRangeV28` and `trainRangeV29` from identical initial weights and requires exact equality of:

- accepted update count;
- target-reached status;
- final cost, mean percentage error and maximum percentage error;
- all 192 final parameter doubles.

The synthetic teacher targets are generated with the same AVX-512/libmvec production forward arithmetic, giving the benchmark an exactly representable optimum. A case fails if it reaches the target with zero updates, fails to reach the target, or changes the v28 trajectory.

Timing pairs alternate execution order. The runner writes a CSV containing median, p10, p90, wins and update count for every row-count/target combination.

## Release interpretation

Evaluator-level evidence is already strong and positive, including real-Wine-derived replay stress. The full-training matrix above remains the final unexecuted production release gate in the current research environment because the complete historical include chain is not locally materialised and this repository does not start GitHub Actions runs.

Do not infer that the full-training matrix passed from the evaluator results alone. Once it has been run on a clone, retain the generated CSV in this directory and review the ten cases before merging v29.
