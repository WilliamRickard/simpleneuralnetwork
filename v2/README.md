# Simple Neural Network v2

This directory contains an optimised companion to the original 2017 `main.cpp`. The original source remains unchanged at the repository root so the two versions can be compared directly.

V2 deliberately stays as a single C++ source file, matching the style of the original project. The aim is to make the performance changes easy to read from top to bottom without introducing a larger project structure.

## What v2 preserves

The network mathematics remain the same as the original batch-online model:

```text
inputs -> W1 -> sigmoid hidden layer -> W2 -> sigmoid output
```

The defaults remain 11 input variables, 16 hidden nodes and one output. V2 still has no bias parameters, uses sigmoid in both layers, uses the same summed squared-error gradient, and uses the same classical momentum update:

```text
velocity = momentum * velocity - learningRate * gradient
weight   += velocity
```

The text-file formats are retained:

- `InputVariables.txt`: whitespace-delimited input rows.
- `OutputVariables.txt`: one target per observation.
- `wone.txt`: W1 stored as `inputs x hidden`.
- `wtwo.txt`: one W2 value per line.
- `ybar.txt`: one prediction per line.

## Source layout

```text
v2/
    main.cpp
    README.md
```

`main.cpp` contains the complete program: configuration, contiguous matrix storage, file I/O, forward propagation, diagnostics, backpropagation, momentum updates and the run modes.

## Main performance changes

V1 uses `vector<vector<double>>`, which allocates each row separately. V2 stores matrix values in one contiguous row-major `vector<double>`, improving cache locality and reducing allocations.

V1 materialises `X^T`, the hidden activation transpose and `W2^T`. V2 applies the backpropagation equations directly and therefore avoids those copies.

For sigmoid activation `a = sigmoid(z)`, the derivative is `a * (1 - a)`. V2 reconstructs this derivative from the activation already in memory instead of storing separate derivative matrices.

The backward pass calculates `delta3`, the W2 gradient, the hidden-layer delta and the W1 gradient in one structured pass over each observation. Only one hidden-sized `deltaTwo` scratch vector is required rather than a full examples-by-hidden matrix.

Cost, mean percentage error and maximum percentage error are calculated while the predictions and targets are already being read. The momentum equation is also applied in one pass over each weight array.

The full data set is loaded once. Sequential batches use row offsets rather than copying observations into temporary matrices. Progress output defaults to once every 1,000 updates rather than every update, and `\n` is used instead of `std::endl` so logging does not force a stream flush each time.

## Small correctness and robustness fixes

`numberOfDescents` is now a genuine maximum number of weight updates. In v1, exceeding the value changes `cost`, while the loop condition depends on `percentageError`, so the intended limit does not terminate the loop.

The offline path also has a finite descent limit. Its legacy update rule remains equivalent to learning rate 1 and momentum 0.

The sigmoid implementation is algebraically equivalent to v1 but avoids an `exp()` overflow for very large negative inputs.

## Building

V2 uses only the C++ standard library and is compatible with C++11 or later. From the repository root:

```text
g++ -std=c++11 -O3 -Wall -Wextra -Wpedantic v2/main.cpp -o simple_nn_v2
```

## Running

The run mode and model dimensions remain near the top of `main()`, as in the original program. The checked-in defaults match the historical configuration:

```text
batchOnline = false
offline     = false
test        = true
inputs      = 11
hidden      = 16
test rows   = 13853
```

The small public `InputVariables.txt` and `OutputVariables.txt` files are toy replacements for the sensitive training data originally used with the project. They do not match those historical dimensions, so the settings must be adjusted to use the public example.

## Reading v1 and v2 together

V1 expresses backpropagation as a sequence of general matrix helper functions. V2 expresses the same equations directly in loops specialised to this network. Keeping both versions as single files makes that before-and-after comparison straightforward.
