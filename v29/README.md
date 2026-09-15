# v29: exact phased AVX-512 evaluator candidate

V29 is an experimental successor to v28. It keeps the v28 arithmetic and optimiser policy but amortises evaluator state across several eight-row tiles.

## Motivation

V28 removed most of the expensive compiler spill/reload traffic around external libmvec `exp` calls, but it still loads and stores all 24 gradient ZMM accumulators once per eight-row tile. It also computes each row's output dot product immediately after hidden sigmoid calls, so the output-layer weight vectors cannot remain live across rows.

V29 groups up to 12 tiles, or 96 rows, and separates independent work into four phases:

1. first-layer forward calculations and hidden sigmoids;
2. output dot products with `wTwo` resident across the group;
3. output sigmoids and metric accumulation in the original tile order;
4. one gradient-state load, backpropagation of every row in the original order, then one gradient-state store.

The explicit scratch for a full 96-row group is about 14 KiB per evaluator thread: 12 KiB of hidden activations plus output and delta arrays.

## Exactness argument

No mathematical approximation is introduced.

- Each first-layer preactivation receives the same FMA sequence as v28.
- Hidden and output sigmoids call the same libmvec/scalar functions with the same inputs.
- Output dot products use the same vector multiply/add and reduction expression.
- Squared-error and percentage-error reductions are applied in the same eight-row tile order.
- Every gradient accumulator receives exactly the same row-order FMA sequence.
- The removed inter-tile gradient stores and reloads are exact double round-trips.
- Fewer-than-eight-row tails delegate directly to `evaluateSliceV28`.

A standalone production-style stress harness passed 425 worker configurations / 1,275 individual slice comparisons covering five seeds, 17 awkward row counts through 100,000, and one through five worker partitions. The in-repository benchmark directly includes `v29/main.cpp` so the same check can be run against the committed production kernel.

## Initial performance evidence

Adjacent alternating-order paired evaluator measurements on the benchmark host gave one-worker median speedups of approximately:

- 5k: 1.171x
- 10k: 1.170x
- 20k: 1.158x
- 50k: 1.188x
- 100k: 1.194x

All five one-worker cases won at least 24/25 pairs. Three-worker medians remained positive, approximately 1.12x-1.21x, but short cases had wider tails. Four-worker tests remained positive in median but are contaminated by the host's known four-CPU cgroup saturation.

Standalone assembly inspection also reduced the evaluator function from roughly 3.5 KiB to 2.5 KiB and reduced observed ZMM stack references from 42 to 23.

These are candidate-level evaluator results, not yet a v29 release claim. Full-optimiser and Wine validation remain release gates.

## Scope

The v29 optimiser wrapper preserves:

- v28 thread selection and deterministic partial reduction;
- v27/v28 Gauss-Newton behaviour on both sides of the 50k boundary;
- v26 dense BFGS operations;
- v24 line-search mathematics;
- stopping rules and final scalar confirmation.

Only the repeated deep-target evaluator and its Armijo dispatch use the v29 kernel.
