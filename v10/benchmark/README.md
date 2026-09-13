# V10 benchmark notes

This directory contains the final exact-frontier comparison between v9 and v10.

## Retained workload

The only new production dispatch is:

- 1,000,000 or more rows
- exactly 4 threads
- AVX-512/libmvec available

All other paths delegate directly to v9 and therefore have no v10 arithmetic or performance change.

## Release method

The release timing uses 1,000,000 rows, 100 updates and five alternating v9/v10 pairs. Four threads are pinned to CPUs 0-3 with `OMP_PROC_BIND=close` and `OMP_PLACES=cores`.

`benchmark_summary.csv` reports both the conventional ratio of separate medians and the more conservative median of the paired ratios.

## Exactness

The development harness compares every byte of W1, W2, momentum W1 and momentum W2 after each timed run. All five release pairs were byte-identical.

This is stronger than checksum equality. The checksum is retained in `raw_timings.csv` as an additional quick diagnostic.

## Screening

The retained eight-observation forward kernel was also screened at other thread counts. Two and five threads regressed. Three threads showed only a small short-run gain. Single-thread workloads regressed. Production therefore selects v10 only for the four-thread large-batch path.

Gradient-splitting and compiler-build experiments are documented in `experiments.txt`.
