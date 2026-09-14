# v25: adaptive AVX-512 evaluator parallelism

V25 keeps v24's optimiser mathematics, metric-balanced dense seed, 0.015% dense-BFGS handover and safeguarded dense-tail quadratic Armijo search. The retained change is the worker-selection policy used by the hot objective/gradient evaluator for deep targets.

## Retained policy

For targets above 0.001%, v25 delegates to v24 unchanged.

For targets at or below 0.001%, v25 keeps the v24 optimisation logic but replaces the evaluator's inherited v15 worker policy. V15 forced every evaluation below 50,000 rows to one thread. On the current AVX-512 evaluator this leaves substantial parallelism unused.

V25 chooses workers as follows:

```text
requested = caller thread request, or 4 when request is 0
capped    = clamp(requested, 1, 4)
useful    = max(1, rows / 1024)
threads   = min(capped, useful)
```

The existing four-worker cap is retained. A caller requesting one thread still gets one thread. Small batches remain serial: for example, the 133-row Wine validation set is unchanged. Workloads of 4,096 rows or more can use all four workers when requested.

The Gauss-Newton diagonal refresh remains on the inherited policy. It occurs only a small number of times per run and is not the runtime hotspot. V25 changes the repeated objective/gradient evaluations that dominate runtime.

## Why this helps

`evaluateV17` is the dominant runtime cost. The old 50k cutoff dates from v15, before the current AVX-512 evaluator and the later deep-convergence optimiser work. Measurements on the current host show that the AVX-512 row kernel has enough work per evaluation for four workers to pay off far below 50k rows.

Parallel reduction changes floating-point grouping. Therefore v25 is not expected to reproduce v24's exact update or row-pass counts on 4k–49,999-row workloads. The release criterion is target attainment and measured wall-clock improvement, not bit-identical trajectories.

## Paired timing results

The table below uses rotating-order paired measurements. Times are optimiser-only medians from the AVX-512 research path with the same v24 optimiser policy. The 50k and 100k production workloads already used four threads in v24 and are intentionally unchanged by v25.

| Dataset | Rows | Target | v24 workers | v25 workers | v24 median | v25 median | Median speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic | 5,000 | 0.001% | 1 | 4 | 0.2735 s | **0.0765 s** | **3.57x** |
| Synthetic | 5,000 | 0.0005% | 1 | 4 | 0.4530 s | **0.1657 s** | **2.73x** |
| Synthetic | 10,000 | 0.001% | 1 | 4 | 0.3414 s | **0.1418 s** | **2.41x** |
| Synthetic | 10,000 | 0.0005% | 1 | 4 | 0.5924 s | **0.2464 s** | **2.40x** |
| Synthetic | 20,000 | 0.001% | 1 | 4 | 0.4751 s | **0.1846 s** | **2.57x** |
| Synthetic | 20,000 | 0.0005% | 1 | 4 | 1.0222 s | **0.3653 s** | **2.80x** |
| Wine x100 timing stress | 13,300 | 0.001% | 1 | 4 | 1.1856 s | **0.3921 s** | **3.02x** |
| Wine x100 timing stress | 13,300 | 0.0005% | 1 | 4 | 1.1585 s | **0.3399 s** | **3.41x** |

The 13,300-row Wine timing stress repeats the same 133 real training observations 100 times. It adds no synthetic feature values and exists only to make evaluator timing long enough to distinguish from scheduler noise. V25 won all 5/5 paired runs at both targets in that test.

Pass counts are included in `benchmark/thread_policy_summary.csv` for transparency. They can move in either direction because changing the reduction partition changes floating-point rounding and therefore the optimisation trajectory. For example, 20k / 0.0005% changed from 1,183 to 1,192 passes while still becoming about 2.8x faster.

## Real-data quality checks

The original 133-row Wine training split stays on one worker under the retained 1,024-rows-per-worker policy, so v25 does not change that small-data trajectory at all.

As an additional sensitivity test, forcing four workers across 10 stratified Wine splits did not show a systematic quality loss: mean hold-out accuracy was 78.22% versus 76.89% for one worker, with four accuracy wins, four ties and two losses. This is a trajectory-sensitivity check, not evidence that threading improves predictive quality.

## Rejected evaluator ideas

Several alternatives were screened before retaining the worker-policy change:

- AVX-512 reciprocal approximation plus two Newton refinements preserved the evaluator to roughly double precision but was slower than hardware division once the full sigmoid path was measured.
- Replacing the per-block maximum-error store/scan with `_mm512_reduce_max_pd` did not give a consistent speedup.
- Replacing `exp` with `exp2(x * log2(e))` was faster in isolated evaluator tests but changed deep optimiser trajectories enough to produce mixed end-to-end results. It is not retained.

## Validation

The exact v25 production wrapper has passed strict local C++11 portable and accelerated compile gates against the inherited v24 interface:

```text
-O3 -Wall -Wextra -Wpedantic -Werror
-O3 -Wall -Wextra -Wpedantic -Werror -fopenmp -DSIMPLE_NN_USE_LIBMVEC -lmvec
```

The compile check uses the same inherited-interface stub method as prior releases. GitHub Actions status must be checked separately and is not implied by local compilation.
