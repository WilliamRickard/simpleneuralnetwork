# V12 benchmark notes

V12 compares the v11 FP64 adaptive-sigmoid training kernel against FP64-master/FP32-compute mixed precision, and separately screens a full-FP32 master-weight variant.

## Release matrix

- 100,000 rows x 100 updates, 1 thread, 7 paired repetitions
- 500,000 rows x 40 updates, 1 thread, 7 paired repetitions
- 1,000,000 rows x 30 updates, 1 thread, 7 paired repetitions
- 1,000,000 rows x 60 updates, 4 threads, 5 paired repetitions

The v11 reference uses four-row forward grouping for single-thread cells and eight-row grouping for the four-thread million-row cell, matching v11 production dispatch. The retained v12 kernel uses 16-wide FP32 hidden vectors and four-row forward grouping.

`benchmark_summary.csv` contains medians, paired reductions and final-weight drift. `raw_timings.csv` contains every retained v11/v12 timing pair.

## Long-run checks

- 100,000 rows x 1,000 updates, 1 thread
- 1,000,000 rows x 300 updates, 4 threads

The mixed path remained within `1.33e-10` and `4.50e-11` maximum weight difference respectively on these tests.

## Rejected full-FP32 candidate

Full FP32 master weights and momentum did not materially improve runtime over the mixed design. It increased final-weight drift to roughly `1e-7` in the release matrix and `1e-6` to `3e-6` in longer tests, so it is recorded as an experiment rather than retained in production.

## Host

- AMD EPYC 9V74
- GCC 14.2.0
- Linux x86-64
- AVX-512 / AVX512_BF16 available
- glibc libmvec
