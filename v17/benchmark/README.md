# V17 benchmark methodology

The release comparison uses the same deterministic 11 -> 16 -> 1 sigmoid problem as v14-v16.

Each workload runs v16 and v17 in alternating order for seven repetitions. Both implementations use four workers and the same L-BFGS controls. Timing starts before the initial objective/gradient evaluation and ends after the final scalar confirmation pass, so the reported numbers are production-shaped end-to-end time-to-target measurements rather than evaluator-only timings.

V16 confirmation is the inherited serial scalar pass. V17 computes the same scalar `predictRow` function independently across four workers, stores the predictions, then aggregates squared and percentage errors in row order.

The three release workloads are 100,000, 500,000 and 1,000,000 rows. All retained runs reach the 3.9% target after three L-BFGS evaluations.

See `benchmark_summary.csv` and `raw_timings.csv` for the retained evidence, and `experiments.txt` for rejected alternatives.
