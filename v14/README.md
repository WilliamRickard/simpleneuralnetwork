# Simple Neural Network v14

V14 changes the optimisation algorithm rather than reducing arithmetic precision further. It keeps the existing 11 -> 16 -> 1 sigmoid network and exact FP64 objective/gradient calculation, but replaces fixed-learning-rate full-batch momentum with limited-memory BFGS (L-BFGS) and Armijo backtracking.

## Retained algorithm

- 192 trainable parameters are packed into one optimisation vector.
- Full-batch objective and gradient are evaluated through the inherited FP64 calculation.
- L-BFGS keeps the most recent 10 `(s,y)` curvature pairs.
- The inverse-Hessian action uses the standard two-loop recursion.
- Each iteration starts with step size 1 and uses Armijo backtracking with `c1 = 1e-4`.
- A non-descent direction resets the history and falls back to steepest descent.
- The project percentage-error target remains the stopping condition.
- Final metrics are recomputed through the inherited exact inference path.

`maxDescents` is interpreted as the maximum number of accepted L-BFGS iterations in v14. Momentum and learning-rate settings are intentionally unused by the v14 optimiser.

## Why this is a larger change

V1-v13 mostly reduced the cost of one training pass. V14 optimises the number of passes required to reach the requested result. For this network there are only 192 parameters and the full-batch objective is smooth, making quasi-Newton curvature information unusually effective.

## Time to target

The deterministic benchmark uses the same synthetic 11 -> 16 -> 1 problem used for recent performance work and a percentage-error target of 3.9%.

| Rows | Initial error | Evaluation 2 | Evaluation 3 | Evaluations to target | Median wall time, 5 runs |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 26.680% | 20.077% | **3.069%** | **3** | **0.127 s** |
| 500,000 | 26.683% | 20.080% | **3.066%** | **3** | **0.630 s** |
| 1,000,000 | 26.683% | 20.080% | **3.068%** | **3** | **1.256 s** |

The benchmark harness is an independent scalar C++ reproduction of the v14 L-BFGS equations. Production uses the repository's existing optimised FP64 objective/gradient evaluator, so the benchmark wall times are conservative for the optimiser itself rather than a promise for every production dataset.

## Comparison with other optimiser screens

On the 20,000-row deterministic screen:

- historical full-batch momentum was still at about 14.85% error after 6,000 complete dataset passes;
- a tuned deterministic mini-batch momentum variant (batch 256, mean-gradient learning rate 0.003) reached 3.75% after about 10 dataset-equivalent passes;
- L-BFGS reached about 3.04% on its third full objective/gradient evaluation.

Mini-batching is therefore a real improvement over the historical optimiser, but L-BFGS was substantially better for this small smooth parameter problem and is the retained v14 design.

## Numerical contract

Unlike v11-v13, v14 does not need lower arithmetic precision to obtain its principal speed-up. The optimisation trajectory and final weights differ because the optimiser changes, but each v14 objective/gradient evaluation uses the inherited FP64 calculation. The result should be judged by target achievement and final exact metrics rather than weight equality to momentum training.

## Build

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v14/main.cpp -o simple_nn_v14
```

OpenMP may still be enabled for inherited code, although the retained L-BFGS objective evaluator currently requests the established single-thread full-batch gradient calculation:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp \
    -DSIMPLE_NN_USE_LIBMVEC v14/main.cpp -lm -o simple_nn_v14
```

## Next steps

The main remaining opportunity is no longer lower precision. It is to accelerate each L-BFGS objective/gradient evaluation using the v12/v13 FP32/BF16 kernels while retaining an exact acceptance/target check, or to add a parallel exact gradient evaluator. Either should be benchmarked against v14's three-evaluation baseline rather than against momentum updates.
