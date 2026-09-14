# v28: exact call-boundary-aware AVX-512 evaluator

V28 keeps the v27 optimiser, line-search policy, dense BFGS algebra, Gauss-Newton policy, stopping rules and final confirmation behaviour. It changes only the repeated AVX-512 evaluator used by deep-target training.

## Why this exists

The v17 evaluator keeps 22 first-layer gradient ZMM accumulators live while it calls glibc/libmvec `exp` repeatedly for hidden and output sigmoids. Under the x86-64 SysV ABI, vector registers are caller-saved across those external calls. Assembly inspection showed substantial spill/reload traffic around each vector exponential.

Exploratory assembly on the benchmark host showed approximately:

- 327 stack references in the production-style baseline slice;
- 265 after making the libmvec call boundary explicit;
- 274 ZMM stack references in the baseline;
- 191 after the change.

That is about a 30% reduction in vector spill traffic.

## Retained implementation

For each eight-row tile, v28:

1. performs the first-layer forward FMAs in exactly the v17 order;
2. stores completed preactivations before entering libmvec;
3. runs the same libmvec sigmoid calls and stores hidden activations;
4. runs the same output sigmoid;
5. only after the tile's final `exp`, loads the gradient accumulators;
6. performs the complete eight-row backpropagation in the original row/FMA order;
7. stores the completed gradient state once before the next tile.

The tail path follows the same principle.

No approximation is introduced. Each individual gradient accumulator receives the same FMA sequence as v17.

## Exactness

A fresh standalone benchmark of the production kernel passed 425/425 comparisons covering:

- row counts from 1 through 100,000, including non-multiples of eight;
- one through five worker partitions;
- five deterministic data/weight seeds;
- squared error, mean percentage-error sum, max percentage error and all 192 gradient doubles.

Earlier wider stress testing passed 756/756 cases.

Full optimiser tests at 5k, 10k, 20k, 50k and 100k rows retained identical accepted updates, evaluator calls, rejected line-search trials, percentage error, objective and all 192 final parameter bits at the deep targets.

Real Wine stress validation retained identical final weights, hold-out RMSE and 84.44% classification accuracy.

## Performance

The benchmark host has a four-CPU cgroup quota and full-training timings are noisy under sustained four-worker load. V28 therefore separates the strong kernel-level result from conservative end-to-end interpretation.

Fresh production-kernel validation on the current host:

- 100k, one worker: about 1.059x median evaluator speedup.

Earlier cooled single-worker evaluator medians ranged from about 1.06x to 1.16x across 5k-100k. Three-worker tests with quota headroom were positive at every tested size, typically mid-single-digit to high-single-digit gains.

Wine full-training validation over seven cooled rotating-order pairs gave approximately:

- target 0.001%: 1.105x median speedup;
- target 0.0005%: 1.078x median speedup.

The conservative expectation is therefore a high-single-digit whole-training improvement on this workload family, not the larger peaks observed during noisy four-worker runs.

## Rejected v28 experiments

### Approximate sigmoid

A high-accuracy range-reduced AVX-512 sigmoid accelerated a full evaluator pass by roughly 1.30-1.36x and had observed relative error below about 9e-13 over a wide test range. It was rejected because the tiny perturbation changed the Wine optimisation basin and reduced hold-out classification accuracy from about 84.4% to 71.1%.

### `noplt` libmvec calls

Bit-exact but effectively neutral. External symbol dispatch was not the dominant cost.

### Restricted-domain copy of glibc vector `exp`

Rejected as a poor maintenance/licensing trade-off. glibc's normal AVX-512 path is already aggressively range-reduced and polynomial-based, so removing exceptional-range machinery did not offer enough expected upside.

### Hybrid live-register/L1 gradients

An exact intermediate design was faster than v27 but remained more sensitive to register allocation and cgroup noise than the retained explicit-call-boundary layout.

### Row-major immediate spilled-gradient update

Exact, but repeated gradient loads/stores removed the benefit.

## v27 compatibility

V28 preserves v27's Gauss-Newton policy exactly:

- below 50,000 rows, v27's exact block-parallel GN preparation is retained;
- at 50,000 rows and above, the inherited v26/v20 threaded GN grouping is retained.

Only the repeated evaluator and its Armijo calls use the v28 kernel.
