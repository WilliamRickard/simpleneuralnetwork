# V23 paired timing benchmark

This benchmark measures wall-clock runtime for the deterministic 11-16-1 research problem used by the v21-v23 release screens.

## Method

- Variants: v21, v22, v23.
- Row counts: 5,000, 20,000, 100,000.
- Targets: 0.001% and 0.0005% percentage error.
- Repetitions: 10 measured repetitions per variant/case.
- Warm-up: one discarded run per variant/case before measured repetitions.
- Ordering: variants rotate across repetitions so v21, v22 and v23 are measured adjacent in time rather than in long version-specific blocks.
- Threads: 5 deterministic OpenMP worker slices.
- Build: `g++ 14.2.0`, C++11, `-O3 -mavx2 -mfma -fopenmp -Wall -Wextra -Wpedantic -Werror -lmvec`, without `-ffast-math`.
- Host: Intel Xeon Platinum 8573C, 5 visible cores, one hardware thread per visible core.
- Clock: `std::chrono::steady_clock`.
- `optimizer_s` starts immediately before the initial objective/gradient evaluation and ends after convergence.
- `data_s` measures deterministic dataset construction.
- `end_to_end_s` includes dataset construction plus optimizer work inside the benchmark process.
- p10/p90 use linear interpolation over the 10 measured observations.
- Raw timings are retained. No slow observations were dropped.
- All 180 measured runs reproduced the expected deterministic update/evaluation/Gauss-Newton counts for their variant/case.

The host is shared and not CPU-isolated. Wall-clock timings therefore have environmental noise. The paired rotating order is intended to reduce bias from host-load drift, but the deterministic row-pass counts remain the stronger release metric.

## Optimizer timing results

Values are median seconds with `[p10, p90]`. Speedups are medians of per-repetition paired speedup ratios, not ratios of the displayed medians.

| Rows | Target | v21 optimizer s | v22 optimizer s | v23 optimizer s | v23 vs v22 | v23 vs v21 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 0.0005% | 0.718 [0.679, 0.827] | 0.392 [0.376, 0.636] | 0.283 [0.263, 0.343] | 1.42x | 2.50x |
| 5,000 | 0.0010% | 0.358 [0.336, 0.429] | 0.234 [0.195, 0.249] | 0.140 [0.121, 0.149] | 1.71x | 2.63x |
| 20,000 | 0.0005% | 3.289 [3.197, 3.724] | 1.341 [1.256, 2.245] | 1.045 [1.005, 1.066] | 1.30x | 3.12x |
| 20,000 | 0.0010% | 1.020 [0.976, 1.190] | 0.763 [0.710, 1.034] | 0.512 [0.468, 0.566] | 1.48x | 2.04x |
| 100,000 | 0.0005% | 9.453 [9.207, 11.423] | 6.822 [6.367, 7.252] | 3.548 [3.413, 4.088] | 1.88x | 2.67x |
| 100,000 | 0.0010% | 3.717 [3.540, 4.047] | 2.908 [2.874, 3.472] | 1.991 [1.935, 2.135] | 1.49x | 1.85x |

## Interpretation

V23 is faster than v22 in every retained case. Its paired median speedup over v22 ranges from 1.30x to 1.88x, and over v21 from 1.85x to 3.12x.

The largest retained calculation, 100,000 rows to 0.0005% PE, has median optimizer times of 9.453s for v21, 6.822s for v22, and 3.548s for v23.

See:
- `timing_raw.csv` for all 180 measurements.
- `timing_summary.csv` for per-variant p10/median/p90 summaries.
- `timing_speedups.csv` for paired speedup distributions.
- `timing_benchmark.cpp` for the instrumented reproducibility harness.
