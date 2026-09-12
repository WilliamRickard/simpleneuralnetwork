# Simple Neural Network v2

This directory contains an optimised companion to the original 2017
`main.cpp`. The original source remains unchanged at the repository root so the
two versions can be compared directly.

## What v2 preserves

The network mathematics remain the same as the original batch-online model:

```text
inputs -> W1 -> sigmoid hidden layer -> W2 -> sigmoid output
```

The defaults remain 11 input variables, 16 hidden nodes and one output. V2
still has no bias parameters, uses sigmoid in both layers, uses the same summed
squared-error gradient, and uses the same classical momentum update:

```text
velocity = momentum * velocity - learningRate * gradient
weight   += velocity
```

The text-file formats are also retained:

- `InputVariables.txt`: whitespace-delimited input rows.
- `OutputVariables.txt`: one target per observation.
- `wone.txt`: W1 stored as `inputs x hidden`.
- `wtwo.txt`: one W2 value per line.
- `ybar.txt`: one prediction per line.

Weights written by v2 therefore use the same layout as v1.

## Source layout

- `main.cpp`: run modes and historical configuration values.
- `matrix.hpp`: contiguous row-major matrix storage.
- `model.hpp`: network, data, metrics and reusable training buffers.
- `io.hpp`: v1-compatible file input/output and weight initialisation.
- `engine.hpp`: forward propagation, diagnostics, backpropagation and updates.
- `run_modes.hpp`: batch-online, offline and test workflows.

The code is split into small components so the performance changes can be read
without losing the educational connection to the underlying equations.

## Main performance changes

### Contiguous matrix storage

V1 uses `vector<vector<double>>`, which allocates every row separately. V2 uses
one row-major `vector<double>` per matrix. This improves cache locality and
removes a large number of small heap allocations.

### No explicit transposes in training

V1 materialises `X^T`, the hidden activation transpose and `W2^T`. V2 applies
the same backpropagation equations directly, so these matrices and copies are
unnecessary.

### No derivative matrices

For sigmoid activation `a = sigmoid(z)`, the derivative is `a * (1 - a)`. V2
reconstructs the derivative from the activation already in memory instead of
storing separate derivative matrices.

### Fused backpropagation

V2 calculates `delta3`, `gradientW2`, the hidden-layer delta and `gradientW1`
in one structured pass over each observation. It only needs one hidden-sized
`delta2` scratch vector rather than an examples-by-hidden matrix.

### Fused diagnostics

During training, cost, mean percentage error and maximum percentage error are
calculated while backpropagation is already reading predictions and targets.
V1 scans those values in several separate helper functions.

### Fused momentum update

V1 scales momentum, scales gradients, subtracts gradients and adds updates in
separate matrix passes. V2 performs the equivalent equation in one pass over
each weight array.

### Reused buffers and batch views

Scratch arrays are allocated once per batch and reused for every descent. The
full data set is loaded once, and sequential batches are represented by a row
offset rather than copied into temporary X/y matrices.

### Less console I/O

Progress output defaults to once every 1,000 updates instead of every update.
Newlines use `\n` rather than `std::endl`, avoiding a forced stream flush for
each progress message.

## Small correctness and robustness fixes

V2 implements two behaviours that the original comments and README appear to
intend:

1. `numberOfDescents` is now a genuine maximum number of weight updates. In v1,
   exceeding the value changes `cost`, while the loop condition depends on
   `percentageError`, so the limit does not terminate the loop.
2. The offline path has a finite descent limit so an unreachable target cannot
   create an unbounded loop. Its legacy update rule remains equivalent to
   learning rate 1 and momentum 0.

The sigmoid implementation is algebraically equivalent to v1 but avoids an
`exp()` overflow for very large negative inputs.

## Building

V2 uses only the C++ standard library and is compatible with C++11 or later.
From the repository root, one example build command is:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v2/main.cpp -o simple_nn_v2
```

## Running

The run mode and model dimensions remain visible near the top of `main()` so v2
can be experimented with in the same style as v1. The checked-in defaults match
the historical v1 configuration:

```text
batchOnline = false
offline     = false
test        = true
inputs      = 11
hidden      = 16
test rows   = 13853
```

The small public `InputVariables.txt` and `OutputVariables.txt` files are only a
toy replacement for the private training data used when the project was first
written. They do not match those historical dimensions. To run the public toy
data, adjust the row/input settings and provide compatible weight files, just
as with v1.

## Reading v1 and v2 together

V1 expresses backpropagation as a sequence of general matrix helpers. V2
expresses the same equations directly in loops specialised to this network.
That removes intermediate matrices while keeping the mathematics visible, so
the two versions provide a useful before-and-after comparison of implementation
style without replacing the original project.
