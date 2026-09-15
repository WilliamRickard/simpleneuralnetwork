# v29: exact phased AVX-512 evaluator

V29 is the validated successor to v28. It keeps the v28 arithmetic and optimiser policy but amortises evaluator state across several eight-row tiles.

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

The production exactness harness passed 425/425 worker configurations and 1,275/1,275 individual slice comparisons covering five seeds, 17 awkward row counts through 100,000, and one through five worker partitions.

A second randomized differential stress passed 4,000/4,000 additional bit-exact slice comparisons. It varied start offsets, slice lengths, zero targets, feature scales from 0.05 to 50 and network-weight scales from 0.01 to 12, with occasional slices of roughly 5k to 100k rows. Metrics and all 192 gradient doubles matched exactly in every case.

## Performance evidence

Adjacent alternating-order paired evaluator measurements on the benchmark host gave one-worker median speedups of approximately:

- 5k: 1.171x
- 10k: 1.170x
- 20k: 1.158x
- 50k: 1.188x
- 100k: 1.194x

All five one-worker cases won at least 24/25 pairs. Three-worker medians remained positive, approximately 1.12x-1.21x, but short cases had wider tails. Four-worker tests remained positive in median but were contaminated by the host's known four-CPU cgroup saturation.

Standalone assembly inspection also reduced the evaluator function from roughly 3.5 KiB to 2.5 KiB and reduced observed ZMM stack references from 42 to 23.

### Wider-activation stress

To check that the gain was not dependent on the mild synthetic activation range, the same paired evaluator test was repeated with network-weight scales from 0.25 through 64. Across 25 pairs per scale, median speedups stayed positive at roughly 1.13x-1.21x and the v29 result remained bit-exact.

### Fresh Wine-derived evaluator replay

The exact historical v25-v28 Wine preprocessing script is not retained in the repository, so v29 does not claim to reproduce the earlier 84.44% hold-out run from first principles. Instead, a new evaluator-only stress used scikit-learn's bundled UCI Wine data with a deterministic stratified 133/45 split, standardisation on the 133 training rows, the first 11 features to match the fixed network input count, and the 133-row training matrix repeated 100 times to 13,300 rows.

A 64-state deterministic network replay then exercised a broad range of weight scales. All replay states were bit-exact between v28 and v29 before timing. Across 13 alternating-order paired replay measurements:

- one worker: median 1.224x, 13/13 wins;
- three workers: median 1.165x, 12/13 wins;
- four workers: median 1.120x, 9/13 wins, with the known four-CPU quota noise.

This is a fresh evaluator stress rather than a replacement for the historical full-training Wine validation.

### Group-size tuning

A finer 8/10/12/14/16-tile sweep did not justify increasing the production group beyond 12 tiles. Larger groups could win isolated single-worker cases, but 12 tiles was more robust at 50k and 100k under three- and four-worker partitioning. The retained 96-row group also leaves more L1 headroom.

### Full-training release matrix

The final production v28-v29 matrix used two threads, seven alternating-order pairs per case, a 200 ms cooldown outside the timed interval, `OMP_PROC_BIND=true`, `OMP_PLACES=cores` and `OMP_DYNAMIC=false`. It tested 5k, 10k, 20k, 50k and 100k rows at both 0.001% and 0.0005% targets.

All ten cases reached their targets with exact v28-v29 update counts, final metrics and all 192 final parameter doubles. V29 won all 70 timed pairs. Median whole-training speedups ranged from 1.155x to 1.322x; the 50k deep-target case reached 1.302x and the two 100k cases reached 1.262x and 1.322x. Every recorded p10 speedup remained above 1.14x.

The retained release results are in `benchmark/full_training_summary.csv`.

The release executables built from the committed source were byte-identical to the executables exercised on the AVX-512 release host:

- evaluator SHA-256: `93064024d0a63b28a9044583edebdde5ddf8cd53f3ab3ad9a11694f044597e35`
- full-training SHA-256: `c1c283a327b932f82030a0c974539f7fd493210ab6b9b03899eb24c2ec601daa`

## Compatibility and scope

The v18-v27 recursive source inclusion chain now uses explicit per-version no-main guards rather than nested `main` macro renaming. V18 also pre-includes the standard headers that otherwise first appeared inside its namespace boundary. Historical v18-v26 portable and accelerated standalone compatibility builds passed, and v27, v28 and v29 passed strict portable and accelerated C++11 builds with `-Wall -Wextra -Wpedantic -Werror`.

The v29 optimiser wrapper preserves:

- v28 thread selection and deterministic partial reduction;
- v27/v28 Gauss-Newton behaviour on both sides of the 50k boundary;
- v26 dense BFGS operations;
- v24 line-search mathematics;
- stopping rules and final scalar confirmation.

Only the repeated deep-target evaluator and its Armijo dispatch use the v29 kernel.
