# v18: scale-aware L-BFGS curvature

V18 targets the low-error plateau found after v17. The deterministic benchmark can be represented exactly by the existing 11 -> 16 -> 1 network, so the approximately 0.0354% plateau is not a capacity floor.

## Root cause

V17 accepts an L-BFGS curvature pair only when `s.y > 1e-12`. The objective and gradients are batch means. Near convergence, valid positive curvature naturally falls below that absolute threshold, so the history stops refreshing and the optimiser stalls.

V18 replaces the absolute test with the dimensionless scale-aware condition

```
s.y > 1e-8 * ||s|| * ||y||
```

with an explicit positive-curvature requirement. This is invariant to common objective scaling. The v17 FP64 AVX-512 evaluator, four-worker execution, history length, Armijo line search, target semantics and scalar final confirmation are unchanged.

## Screening results

On the deterministic 100k benchmark, extending unmodified v17 from 100 to 500 iterations remained at about 0.035459% error. History sizes 5, 10, 20 and 40, periodic history resets, and Armijo-constant changes did not remove the plateau.

With scale-aware curvature, the same custom L-BFGS reaches approximately:

| Target | Final error | Evaluations |
| ---: | ---: | ---: |
| 3.9% | 3.068918% | 3 |
| 1% | 0.675751% | 12 |
| 0.1% | 0.057811% | 18 |
| 0.05% | 0.039333% | 19 |
| 0.02% | 0.019979% | 492 |
| 0.01% | 0.009983% | 2341 |

The low-error floor is therefore broken without changing the fast ordinary-target trajectory.

At 1m rows, a screening run reached 0.019991% for a 0.02% target. Deep convergence is deliberately much more expensive than the usual 3.9% target.

## Rejected alternatives

- Merely increasing v17 to 500 iterations: still ~0.03546%.
- L-BFGS history sizes 5/20/40: no material improvement.
- Periodic history resets: no material improvement.
- Different Armijo constants: no material improvement.
- Naive strong-Wolfe condition: regressed convergence.
- Safeguarded zoom strong-Wolfe: eventually crossed 0.02%, but required more evaluations than the simpler scale-aware-curvature fix and complicated the fast path.
- Wider hidden layers: unnecessary for this benchmark because the generating architecture is itself 11 -> 16 -> 1 and has exactly zero representational error.

## Numerical contract

V18 does not change the evaluator's precision contract. It changes which mathematically valid positive-curvature pairs are retained at very small gradient scales. Final target confirmation remains the inherited scalar FP64 prediction path.
