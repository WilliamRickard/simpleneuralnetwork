# v19: staged deep-convergence L-BFGS memory

V19 continues the convergence work from v18. V18 removed the first low-error floor by replacing the absolute curvature gate with a scale-aware positive-curvature test. The remaining cost below roughly 0.01% is dominated by the quality of the limited-memory inverse-Hessian approximation.

## Retained change

V19 keeps the v18 optimiser unchanged while percentage error is above 0.05%. Once the current training percentage error is at or below 0.05% and a deeper target still requires another iteration, the optimisation enters a sticky deep stage and permits the L-BFGS history to grow from 10 to 160 curvature pairs.

The following remain unchanged:

- FP64 weights, activations, gradients and deltas;
- the v17 four-worker AVX-512 evaluator;
- v18 scale-aware curvature acceptance;
- Armijo backtracking and its constants;
- the MSE training objective;
- percentage-error stopping semantics;
- scalar FP64 final confirmation.

For a target of 0.05% or above, the deep stage is never entered. The ordinary trajectory therefore follows the v18 history-10 path exactly.

## Why memory is the retained fix

Post-v18 diagnostics showed that valid curvature pairs continue to be accepted in the deep region, gradients remain representable in FP64, and the line search usually accepts a full step. The L-BFGS direction nevertheless becomes poorly aligned with the gradient as history 10 ages out useful curvature information.

Increasing memory after the v18 curvature fix materially improves deep convergence. A staged transition is preferable to using a large history from iteration zero because it preserves the established v18 fast path and only pays the additional two-loop-recursion cost when deep convergence is requested.

History 160 was retained after sensitivity screening. History 192 did not improve the 100k 0.001% screen and required more evaluations than 160. Entering the larger-memory stage at 0.02% or 0.01% was also worse than entering at 0.05%.

## Independent FP64 confirmation

A clean NumPy FP64 implementation of the same deterministic benchmark, MSE gradient, Armijo search, v18 curvature rule and staged history policy was used as an independent algebraic check. Reduction order differs from the production SIMD evaluator, so exact iteration counts are not expected to be bit-identical.

On 20k rows:

| Variant | Target | Accepted updates | Objective/gradient evaluations | Final PE | Final objective |
| --- | ---: | ---: | ---: | ---: | ---: |
| v18-style history 10 | 0.005% | 3000 cap | 4308 | 0.0062326168% | 7.0763873e-10 |
| v19 staged 10 -> 160 | 0.001% | 3064 | 3363 | 0.0009981362% | 1.2538008e-11 |

The staged run rejected no curvature pairs and triggered no non-descent resets. Its fast-path crossings were 3 evaluations to 3.9%, 12 to 1%, 18 to 0.1% and 19 to 0.05%, matching v18.

Earlier C++ screening in the same investigation reached approximately 0.001% on 100k rows with staged history 160 in about 3197 objective/gradient evaluations, with final PE about 0.00099488%. A 1m-row screen reached 0.02% in about 100 evaluations and was around 0.00563% after 400 accepted updates and 0.00529% after 500. The long 1m tail was not used for timing claims because execution-time variability made the available environment unsuitable for a stable full-tail timing comparison.

## Rejected or deferred alternatives

- History 10 remains too weak after the v18 curvature fix at deep targets.
- History 192 does not consistently improve on 160.
- Delaying the deep-memory transition to 0.02% or 0.01% slows the deep tail.
- Strong Wolfe remains deferred because diagnostics do not show Armijo backtracking as the present bottleneck.
- Simple W1/W2 rescaling remains deferred because the measured layer-gradient norms are of the same order in the deep region.
- Curvature-pair filtering or compression remains a plausible future improvement because the 160 stored step vectors become substantially redundant near 0.001%.

## Numerical contract

V19 changes only how many already-valid v18 curvature pairs can remain available after the deep-stage threshold. It does not weaken numerical precision or target confirmation. Final scalar FP64 confirmation remains authoritative.
