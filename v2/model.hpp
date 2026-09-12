#ifndef SIMPLE_NN_V2_MODEL_HPP
#define SIMPLE_NN_V2_MODEL_HPP

#include "matrix.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace simple_nn_v2 {

/** One row of X corresponds to one scalar target in y. */
struct Dataset {
    Matrix x;
    std::vector<double> y;
};

/** Diagnostics retained from the original program. */
struct Metrics {
    double cost = 0.0;
    double percentageError = 0.0;
    double maxPercentageError = 0.0;
};

/**
 * Network parameters plus momentum buffers.
 *
 * W1 has shape [inputs, hidden]. W2 is a vector because the historical model
 * has exactly one output node, so a [hidden, 1] matrix is unnecessary.
 */
struct Network {
    Matrix w1;
    std::vector<double> w2;
    Matrix velocityW1;
    std::vector<double> velocityW2;

    Network(std::size_t inputCount, std::size_t hiddenCount)
        : w1(inputCount, hiddenCount),
          w2(hiddenCount, 0.0),
          velocityW1(inputCount, hiddenCount),
          velocityW2(hiddenCount, 0.0) {}
};

/**
 * Reusable scratch arrays for one training batch.
 *
 * They are allocated once per batch and reused for every gradient descent.
 * In particular, there is no X transpose, hidden transpose, W2 transpose or
 * examples-by-hidden delta-two matrix.
 */
struct TrainingBuffers {
    Matrix hidden;
    std::vector<double> predictions;
    Matrix gradientW1;
    std::vector<double> gradientW2;
    std::vector<double> delta2;

    TrainingBuffers(std::size_t batchSize,
                    std::size_t inputCount,
                    std::size_t hiddenCount)
        : hidden(batchSize, hiddenCount),
          predictions(batchSize, 0.0),
          gradientW1(inputCount, hiddenCount),
          gradientW2(hiddenCount, 0.0),
          delta2(hiddenCount, 0.0) {}
};

/** Settings for the batch-online momentum update. */
struct OptimiserConfig {
    std::size_t maxDescents = 1000000;
    std::size_t logEvery = 1000;
    double percentageErrorTarget = 3.9;
    double learningRate = 0.0001;
    double momentum = 0.75;
};

/** Result from fitting one sequential batch. */
struct TrainResult {
    Metrics metrics;
    std::size_t updates = 0;
    bool reachedTarget = false;
};

/**
 * Numerically stable form of the same sigmoid used by v1.
 *
 * Splitting positive and negative inputs avoids calling exp() with an argument
 * large enough to overflow while leaving the mathematical function unchanged.
 */
inline double sigmoid(double value) noexcept {
    if (value >= 0.0) {
        const double exponent = std::exp(-value);
        return 1.0 / (1.0 + exponent);
    }

    const double exponent = std::exp(value);
    return exponent / (1.0 + exponent);
}

}  // namespace simple_nn_v2

#endif  // SIMPLE_NN_V2_MODEL_HPP
