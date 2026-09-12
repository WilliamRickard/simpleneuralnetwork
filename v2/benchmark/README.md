# V1 vs V2 performance benchmark

These files reproduce and document the benchmark used to compare the original `main.cpp` with `v2/main.cpp`.

## Results

Both versions used the same deterministic 11-input, 16-hidden-node regression problem, identical initial weights, identical learning-rate and momentum equations, fixed update counts, and no console logging inside the timed region. Both benchmark programs were compiled with:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic
```

Dataset generation was outside the training timer. Peak resident memory was measured with `/usr/bin/time -v`.

| Rows | Updates | V1 median | V2 median | Speed-up | Time reduction | V1 peak RAM | V2 peak RAM | RAM reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13,853 | 500 | 2.289 s | 1.172 s | 1.95x | 48.8% | 19.5 MiB | 4.8 MiB | 75.6% |
| 100,000 | 100 | 3.737 s | 1.682 s | 2.22x | 55.0% | 129.9 MiB | 23.8 MiB | 81.7% |
| 500,000 | 20 | 4.838 s | 1.782 s | 2.71x | 63.2% | 642.6 MiB | 112.2 MiB | 82.5% |
| 1,000,000 | 10 | 4.975 s | 1.830 s | 2.72x | 63.2% | 1,283.5 MiB | 222.9 MiB | 82.6% |

The final weight checksum matched between V1 and V2 in every benchmark case. Cost and percentage-error differences were limited to floating-point summation-order noise.

The main result is that V2 scales better as the dataset grows. At one million rows it was about 2.72x faster and used about 5.76x less peak memory.

## Files

- `bench_v1.cpp`: controlled benchmark preserving the original V1 data structures and transpose-heavy training path.
- `bench_v2.cpp`: controlled benchmark preserving the V2 contiguous/fused training path.
- `benchmark_summary.csv`: processed timings, throughput, memory use and numerical checks.
- `raw/`: raw repeated timing output for the larger runs.
- `generate_test_data.py`: deterministic generator for the larger text datasets.
- `data/`: initial weights and a manifest for the generated 100,000-row dataset.

## Test data

The benchmark data are synthetic and contain no original/private training data. The generator creates inputs deterministically and uses a fixed 11 -> 16 -> 1 sigmoid teacher network to create a learnable target.

The full 100,000-row text dataset is about 16 MB uncompressed, so it is generated on demand rather than stored as a large duplicate in Git. This command recreates it:

```text
python3 v2/benchmark/generate_test_data.py 100000 --out v2/benchmark/data/test_data_100k
```

The exact SHA-256 hashes of the generated files are recorded in `data/README.md` so the dataset can be checked byte-for-byte.

The same generator can create the larger benchmark data:

```text
python3 v2/benchmark/generate_test_data.py 500000 --out /tmp/test_data_500k
python3 v2/benchmark/generate_test_data.py 1000000 --out /tmp/test_data_1m
```

`generate_test_data.py` requires NumPy.

## Running the controlled benchmark

Compile both harnesses from the repository root:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v2/benchmark/bench_v1.cpp -o /tmp/bench_v1
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v2/benchmark/bench_v2.cpp -o /tmp/bench_v2
```

Each program takes `rows`, `updates` and `repetitions`:

```text
/tmp/bench_v1 100000 100 7
/tmp/bench_v2 100000 100 7
```

For a peak-memory run on Linux:

```text
/usr/bin/time -v /tmp/bench_v1 1000000 10 1
/usr/bin/time -v /tmp/bench_v2 1000000 10 1
```

## Interpretation

These are controlled training-kernel benchmarks, not timings of the unmodified repository binaries. That distinction is deliberate: V1 and V2 have different logging behaviour and the original V1 `numberOfDescents` logic does not provide a reliable fixed-update stop. The harnesses therefore make the work performed by the two versions comparable.

Absolute timings are specific to the machine used. The relative speed-up and memory scaling are the useful comparisons.
