# V6 benchmark

This directory records the measurements used to compare v6 with v5 and the profile that motivated the changes.

## Pre-v6 profile

A stage-level profile of the v5 AVX-512/FMA training path on 1,000,000 rows showed approximately:

| Stage | 1 thread | 4 threads |
|---|---:|---:|
| Hidden sigmoid / vector exp | 36.4% | 35.5% |
| Hidden forward multiply | 24.4% | 23.0% |
| W1 gradient accumulation | 19.2% | 21.4% |
| Error metrics + deltaThree | 9.0% | 8.4% |
| W2 gradient + deltaTwo | 5.3% | 6.0% |
| Output dot product | 3.0% | 3.0% |
| Output sigmoid | 2.7% | 2.6% |

This shifted the next optimisation target away from generic vectorisation and towards block sizing, backward-pass dataflow and diagnostic overhead.

## Changes benchmarked

V6 keeps v5's runtime-dispatched AVX-512/FMA, AVX2/FMA and scalar kernels. The retained changes are:

- `BLOCK_SIZE` reduced from 64 to 16 after a local sweep;
- AVX-512 `deltaTwo` creation fused into all 11 W1 gradient accumulators, avoiding a write/read cycle through the hidden scratch buffer;
- cost and maximum-percentage-error accumulation skipped on routine descents when only mean percentage error is required for the stopping rule.

A target reciprocal cache was rejected because the small speed gain cost about 8 MiB at one million rows. AOCL-LibM was researched but not included because it was not possible to benchmark the binary package without accepting its download EULA in this environment.

## Method

The workloads and deterministic dataset are the same as earlier benchmark rounds:

| Rows | Updates |
|---:|---:|
| 13,853 | 500 |
| 100,000 | 100 |
| 500,000 | 20 |
| 1,000,000 | 10 |

V5 and v6 were compiled with the same C++11/O3/OpenMP/libmvec settings and run alternately. Seven repetitions were recorded for each version/workload/thread-count combination. Dataset generation and file I/O are outside the timed region.

The four-thread runs used `OMP_PLACES=cores` and `OMP_PROC_BIND=close`.

## Results

| Rows | Updates | v5 single | v6 single | Speed-up | Time reduction | v5 4-thread | v6 4-thread | Speed-up | Time reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13,853 | 500 | 0.274728 s | 0.257103 s | 1.069x | 6.4% | 0.087041 s | 0.078014 s | 1.116x | 10.4% |
| 100,000 | 100 | 0.406697 s | 0.383030 s | 1.062x | 5.8% | 0.133399 s | 0.120859 s | 1.104x | 9.4% |
| 500,000 | 20 | 0.437682 s | 0.389382 s | 1.124x | 11.0% | 0.121556 s | 0.106560 s | 1.141x | 12.3% |
| 1,000,000 | 10 | 0.437137 s | 0.385207 s | 1.135x | 11.9% | 0.165444 s | 0.122753 s | 1.348x | 25.8% |

Exact processed values are in `benchmark_summary.csv`; individual observations are in `raw_timings.csv`.

## Memory

For the 1,000,000-row / 10-update / four-thread case:

```text
v5: 95,488 KiB
v6: 95,760 KiB
```

The ~272 KiB difference is negligible and within normal process/runtime variation, so v6 preserves v5's roughly 93 MiB working set.

## Numerical checks

The 100,000-row one-update production test produced the following identical SHA-256 hashes for v5, v6 AVX-512 and a separately forced v6 AVX2 path:

```text
wone.txt  b69ebe75618cac2fbc65c4ada349492fafe8894fe485018db4f7a8ad6a2b386d
wtwo.txt  cf925865fb69e2dd2dead82e0b38309162cd8b178e5262e6da66bf853dc5850f
ybar.txt  af983f2ca38bd36e2886c3834707e12d77226d70f8d525713f1020f3e76f70f7
```

Every paired benchmark workload also finished on one common printed checksum for each version/thread configuration.
