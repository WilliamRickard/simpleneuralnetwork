#!/usr/bin/env python3
"""Generate deterministic non-sensitive regression data for the neural-network benchmark.

The target is produced by a fixed 11 -> 16 -> 1 sigmoid teacher network, so the
problem is learnable by the architecture used in the repository. The generator
is deterministic and can create datasets much larger than the original private
training set without committing sensitive data.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(rows: int, out: Path) -> None:
    inputs, hidden = 11, 16
    out.mkdir(parents=True, exist_ok=True)

    i = np.arange(1, rows + 1, dtype=np.float64)[:, None]
    k = np.arange(1, inputs + 1, dtype=np.float64)[None, :]
    x = 0.55 * np.sin(0.013 * i * k) + 0.35 * np.cos(0.007 * (i + 2.0) * (k + 1.0))

    kk = np.arange(1, inputs + 1, dtype=np.float64)[:, None]
    j = np.arange(1, hidden + 1, dtype=np.float64)[None, :]
    teacher_w1 = 0.22 * np.sin(0.17 * kk * (j + 1.0))
    teacher_w2 = 0.28 * np.cos(0.31 * np.arange(1, hidden + 1, dtype=np.float64))
    y = sigmoid(sigmoid(x @ teacher_w1) @ teacher_w2)

    initial_w1 = 0.12 * np.sin(0.43 * kk * j)
    initial_w2 = 0.15 * np.cos(0.37 * np.arange(1, hidden + 1, dtype=np.float64))

    np.savetxt(out / 'InputVariables.txt', x, fmt='%.10f')
    np.savetxt(out / 'OutputVariables.txt', y, fmt='%.10f')
    np.savetxt(out / 'wone.txt', initial_w1, fmt='%.10f')
    np.savetxt(out / 'wtwo.txt', initial_w2, fmt='%.10f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rows', type=int, nargs='?', default=100000)
    parser.add_argument('--out', type=Path, default=Path('test_data'))
    args = parser.parse_args()
    generate(args.rows, args.out)
