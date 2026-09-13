# V13 benchmark notes

V13 compares the retained v12 FP64-master / FP32-compute kernel with a BF16 first-layer-forward kernel using FP32 accumulation.

## Release matrix

- 100,000 rows x 100 updates, 1 thread, 7 paired repetitions
- 500,000 rows x 40 updates, 1 thread, 7 paired repetitions
- 1,000,000 rows x 30 updates, 1 thread, 7 paired repetitions

Runs alternate v12-first and v13-first ordering.

The retained v13 kernel uses 16-row forward grouping. Only W1/input multiply operands are BF16; sigmoid, W2 forward, backpropagation and gradients remain FP32.

## Long-run checks

- 100,000 rows x 1,000 updates, 1 thread
- 1,000,000 rows x 300 updates, 1 thread

`accuracy.txt` records exact-sigmoid evaluation drift.

## Rejected four-thread path

The four-thread BF16 path was screened at 1,000,000 rows. Short and longer samples disagreed materially depending on paired versus separate medians, so it was not retained. Production four-thread workloads delegate to v12.

## Host

- AMD EPYC 9V74
- GCC 14.2.0
- Linux x86-64
- AVX-512 / AVX512_BF16
- glibc libmvec
