# V7 benchmark and validation

This directory records the profile and measurements used for v7.

## Benchmark environment

- CPU: Intel Xeon Platinum 8573C
- Exposed cores: 5 physical cores, one thread per core
- Compiler: GCC 14.2.0
- C++ standard: C++11
- Accelerated build: `-O3 -fopenmp` with glibc `libmvec`

The v7 benchmark environment differs from the earlier AMD EPYC benchmark round. The authoritative v7 performance result is therefore the paired v6-versus-v7 comparison in this directory, not a direct comparison of absolute seconds against old CSV files.

## Workload

The deterministic synthetic data, teacher network and initial weights use the same formulas as the earlier benchmark suite. Dataset generation is outside the timed training section.

The four standard cases are:

- 13,853 rows, 500 updates
- 100,000 rows, 100 updates
- 500,000 rows, 20 updates
- 1,000,000 rows, 10 updates

Both one-thread and four-thread configurations were measured. Each configuration has nine repetitions. V6 and v7 execution order alternates by repetition.

## Results

See `benchmark_summary.csv` for processed medians and standard deviations and `raw_timings.csv` for every run.

The 13,853-row four-thread v7 configuration intentionally uses the v6 kernel because the new AVX-512 path did not reliably beat v6 at that very small parallel workload. Therefore the observed timing difference on that row is noise between executions of the same arithmetic kernel.

For workloads that dispatch to the new v7 kernel, the paired medians show about 6-12% lower training time.

## Correctness

`equivalence.txt` records the production-path one-update comparison. The v6-kernel baseline, v7 AVX-512 path and forced AVX2 path produced byte-identical stored weights and predictions.

All paired synthetic benchmark runs also ended on exactly the same printed final-weight checksum.

## Memory

`memory.txt` records the 1,000,000-row, one-update, four-thread peak RSS test. Both v6 and v7 measured 95,616 KiB.

## Profile

`profile_before_v7.txt` contains the stage profile that motivated the changes. Hidden sigmoid and forward multiplication had become the dominant v6 costs, while the fused backpropagation introduced in v6 was no longer the main bottleneck.

## Experiments rejected during v7

- A four-tile superblock gave inconsistent results because the extra live state increased register pressure.
- `sigmoid(x) = 0.5 * (1 + tanh(x / 2))` was mathematically equivalent but much slower with the installed glibc vector math implementation.
- A memory-broadcast FMA microkernel was not retained because each input scalar is reused for two hidden-vector FMAs, making one explicit broadcast a better fit than two scalar memory broadcasts.
- oneMKL, SLEEF and AOCL-LibM were not installed in this execution environment, so no unmeasured third-party vector-math backend was added.
