# v26: exact AVX-512 dense BFGS algebra

V26 keeps v25's evaluator, adaptive worker policy, optimiser stages, metric-balanced dense seed, 0.015% dense handover and safeguarded v24 line search. The retained change accelerates only the 192x192 dense inverse-BFGS algebra after the existing handover.

## Retained implementation

The v22-v25 dense inverse is stored row-major and updated with scalar nested loops. V26 uses a column-major representation once dense BFGS begins. The initial dense matrix is diagonal, so its flat row-major and column-major representations are identical at the handover and after any reset.

The column-major layout lets AVX-512 process eight independent matrix rows at once while each lane still accumulates columns in the exact scalar order `j = 0..191`.

Two details are required for numerical equivalence:

- multiplication and addition remain separate rather than using FMA;
- the AVX-512 dense helpers use function-local `fp-contract=off`.

All scalar dot products, curvature tests and BFGS coefficients remain in the inherited generic code path. This preserves the current floating-point trajectory instead of merely producing a numerically close result.

V26 delegates to v25 when the requested target is above 0.001%, when the accelerated build is unavailable, or when AVX-512 is not supported at runtime.

## Exactness gate

The standalone dense benchmark compares the old row-major scalar implementation with v26's column-major AVX-512 implementation bit-for-bit.

Across the retained benchmark:

- dense direction: zero differing doubles;
- one dense BFGS update: zero differing matrix elements;
- after 100 repeated direction/update cycles: zero differing doubles and zero differing matrix elements.

A production-style end-to-end harness then compared the two dense implementations through the full deep optimiser. For all 5k, 10k, 20k, 50k and 100k cases at 0.001% and 0.0005%, updates, rejected line-search trials, final PE and final objective were identical. `benchmark/exact_dense_summary.csv` records the retained counts.

## Dense-kernel speed

Five local repetitions of the focused benchmark gave these medians:

| Operation | v25 scalar | v26 AVX-512 | Speedup |
| --- | ---: | ---: | ---: |
| Dense direction | 18.257 us | 7.586 us | 2.32x |
| Dense update | 35.618 us | 20.195 us | 1.78x |
| Combined | 53.875 us | 27.781 us | 1.95x |

These are measured dense-kernel timings, not whole-training timings.

## Expected production impact

The research end-to-end optimiser harness uses an AVX2 evaluator, whereas production v25 already uses AVX-512. Switching between AVX2 evaluator work and AVX-512 dense work can distort wall-clock comparisons, so those end-to-end timings are not used as production speed claims.

Instead, `benchmark/exact_dense_summary.csv` combines measured production-style AVX-512 evaluator-pass medians with the measured dense-kernel medians. Because v26 is trajectory-identical, the evaluator and dense-iteration counts are unchanged. The resulting component model estimates approximately:

| Rows | 0.001% | 0.0005% |
| ---: | ---: | ---: |
| 5,000 | 1.15x | 1.17x |
| 10,000 | 1.09x | 1.09x |
| 20,000 | 1.05x | 1.05x |
| 50,000 | 1.02x | 1.02x |
| 100,000 | 1.01x | 1.01x |

This model is deliberately conservative and should be treated as an estimate until exact production AVX-512 end-to-end timing is available.

## Rejected alternatives

Profiling before the retained design found:

- persistent OpenMP regions can save only around 1-2% at 5k and almost nothing at 100k because region-entry overhead is only a few microseconds;
- percentage/max-error bookkeeping was not a stable enough hotspot for a release change;
- a cheap sigmoid timing surrogate showed that transcendental work remains material, but no accurate faster `exp` implementation was available on the current toolchain;
- a first row-major AVX-512 dense implementation used horizontal reductions/FMA and was faster but changed the floating-point trajectory, so it was rejected;
- a first column-major implementation without `fp-contract=off` differed by roughly one ulp in many matrix entries and also was rejected.

## Validation

The exact production wrapper passes strict local C++11 builds in both configurations:

```text
-O3 -Wall -Wextra -Wpedantic -Werror
-O3 -Wall -Wextra -Wpedantic -Werror -fopenmp -DSIMPLE_NN_USE_LIBMVEC -lmvec
```

The focused dense exactness benchmark also compiles under C++11 with the same strict warning policy. GitHub Actions status must be checked separately; local compilation is not a claim that CI ran.
