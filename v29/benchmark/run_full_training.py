#!/usr/bin/env python3
"""Compile and run the v28-v29 production full-training benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import tempfile
from pathlib import Path

ROWS = (5000, 10000, 20000, 50000, 100000)
TARGETS = (0.001, 0.0005)


def percentile_nearest(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * p)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("v29/benchmark/full_training_results.csv"))
    args = parser.parse_args()
    if args.pairs < 1 or args.threads < 1:
        parser.error("pairs and threads must be positive")

    root = Path(__file__).resolve().parents[2]
    source = root / "v29" / "benchmark" / "full_training_benchmark.cpp"
    with tempfile.TemporaryDirectory(prefix="v29-bench-") as temporary:
        binary = Path(temporary) / "v29-full-training"
        compile_command = [
            "g++", "-std=c++11", "-O3", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
            "-fopenmp", "-DSIMPLE_NN_USE_LIBMVEC", str(source), "-lmvec", "-o", str(binary),
        ]
        subprocess.run(compile_command, cwd=root, check=True)

        summaries: list[dict[str, object]] = []
        for rows in ROWS:
            for target in TARGETS:
                command = [str(binary), str(rows), str(target), str(args.pairs), str(args.threads)]
                completed = subprocess.run(command, cwd=root, check=True, text=True, capture_output=True)
                ratios: list[float] = []
                updates: set[int] = set()
                for line in completed.stdout.splitlines():
                    if not line or line.startswith("SKIP"):
                        continue
                    fields = line.split(",")
                    if len(fields) != 8:
                        raise RuntimeError(f"unexpected benchmark output: {line}")
                    updates.add(int(fields[4]))
                    ratios.append(float(fields[7]))
                if not ratios:
                    raise RuntimeError(f"no timing results for rows={rows}, target={target}")
                if len(updates) != 1:
                    raise RuntimeError(f"update count moved across identical runs: {updates}")
                summaries.append({
                    "rows": rows,
                    "target": target,
                    "threads": args.threads,
                    "pairs": len(ratios),
                    "updates": next(iter(updates)),
                    "wins": sum(value > 1.0 for value in ratios),
                    "median_speedup": statistics.median(ratios),
                    "p10_speedup": percentile_nearest(ratios, 0.10),
                    "p90_speedup": percentile_nearest(ratios, 0.90),
                    "trajectory_equal": True,
                })
                summary = summaries[-1]
                print(
                    f"rows={rows:6d} target={target:.4g} "
                    f"median={summary['median_speedup']:.4f}x "
                    f"p10={summary['p10_speedup']:.4f} "
                    f"p90={summary['p90_speedup']:.4f} "
                    f"wins={summary['wins']}/{summary['pairs']} "
                    f"updates={summary['updates']}"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
