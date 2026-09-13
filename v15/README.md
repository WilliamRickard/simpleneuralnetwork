# Simple Neural Network v15

V15 accelerates the v14 L-BFGS optimiser by parallelising each exact FP64 full-batch objective/gradient evaluation. The optimiser itself is unchanged: same 10-pair history, two-loop recursion, Armijo backtracking, 3.9% percentage-error target and exact final metric check.

## Retained design

- FP64 objective and gradients throughout.
- Up to four OpenMP workers, with deterministic contiguous row ownership.
- Per-thread FP64 gradients and long-double metric accumulators.
- Thread-local results are reduced in fixed thread order.
- Workloads below 50,000 rows use one evaluator thread.
- Requested thread counts above four are capped at four on the benchmarked path.
- Portable builds without OpenMP remain single-threaded.

Unlike v11-v13, v15 introduces no additional precision relaxation. It changes only the summation/reduction order between row partitions.

## Why four threads

On the five-core benchmark host, a 500,000-row thread sweep produced median time-to-target values of roughly:

| Evaluator threads | Median solve time |
|---:|---:|
| 1 | 0.372 s |
| 2 | 0.197 s |
| 4 | **0.182 s** |
| 5 | 0.187 s |

Four workers were therefore retained rather than blindly using every visible core.

## Time to target versus v14

Seven alternating paired runs per workload, same standalone C++ objective and L-BFGS implementation:

| Rows | v14 1-thread median | v15 4-thread median | Reduction from medians | Median paired reduction |
|---:|---:|---:|---:|---:|
| 100,000 | 0.0774 s | 0.0387 s | **49.9%** | **50.9%** |
| 500,000 | 0.3749 s | 0.1710 s | **54.4%** | **54.3%** |
| 1,000,000 | 0.7408 s | 0.2981 s | **59.8%** | **59.7%** |

Both v14 and v15 reached the target on the third exact objective/gradient evaluation in every retained run.

Final percentage errors were effectively identical. At one million rows the retained v15 result was `3.067909440176633%` versus `3.067909440176705%` for the one-thread baseline.

## Rejected hybrid proposal path

A second candidate used FP32 gradients to propose the L-BFGS direction while retaining FP64 Armijo acceptance and target checks. It was rejected because v14 already converges in only three exact evaluations. The additional proposal evaluations duplicated work and made end-to-end time worse.

At one million rows in the screening harness:

- four-thread exact-only L-BFGS: about `0.357 s` median;
- four-thread FP32-proposal + FP64 acceptance: about `0.711 s` median.

The hybrid path still reached the same target, but roughly doubled solve time.

## Build

Portable:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v15/main.cpp -o simple_nn_v15
```

OpenMP:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic -fopenmp v15/main.cpp -o simple_nn_v15
```

The default v15 training configuration uses four evaluator threads when OpenMP is available, otherwise one.

## Interpretation

V14 supplied the large algorithmic gain by cutting the solve to three evaluations. V15 attacks the remaining cost directly by evaluating those three passes in parallel. The next useful work should focus on either faster exact vectorised evaluation inside each worker or reducing memory/cache overhead, because adding approximate proposal passes did not help once L-BFGS was already this evaluation-efficient.
