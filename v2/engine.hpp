#ifndef SIMPLE_NN_V2_ENGINE_HPP
#define SIMPLE_NN_V2_ENGINE_HPP

#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace simple_nn_v2 {

/**
 * Forward propagation over a contiguous row range.
 *
 * Equations preserved from v1:
 *
 *   hidden = sigmoid(X * W1)
 *   yHat   = sigmoid(hidden * W2)
 *
 * The implementation writes directly into the activated hidden matrix, so
 * separate Z2 and Z3 matrices are unnecessary.
 */
inline void forwardRange(const Dataset& data,
                         std::size_t startRow,
                         const Network& network,
                         TrainingBuffers& buffers) {
    const std::size_t inputCount = network.w1.rows();
    const std::size_t hiddenCount = network.w1.cols();
    const std::size_t batchSize = buffers.hidden.rows();

    for (std::size_t localRow = 0; localRow < batchSize; ++localRow) {
        const std::size_t dataRow = startRow + localRow;
        const double* xRow = data.x.rowData(dataRow);
        double* hiddenRow = buffers.hidden.rowData(localRow);
        std::fill(hiddenRow, hiddenRow + hiddenCount, 0.0);

        // j is the inner loop so both W1 and hidden are traversed contiguously.
        for (std::size_t input = 0; input < inputCount; ++input) {
            const double xValue = xRow[input];
            const double* weightRow = network.w1.rowData(input);
            for (std::size_t hidden = 0; hidden < hiddenCount; ++hidden) {
                hiddenRow[hidden] += xValue * weightRow[hidden];
            }
        }

        double outputPreActivation = 0.0;
        for (std::size_t hidden = 0; hidden < hiddenCount; ++hidden) {
            hiddenRow[hidden] = sigmoid(hiddenRow[hidden]);
            outputPreActivation += hiddenRow[hidden] * network.w2[hidden];
        }
        buffers.predictions[localRow] = sigmoid(outputPreActivation);
    }
}

/** Calculate diagnostics only, used by inference where gradients are unwanted. */
inline Metrics calculateMetrics(const Dataset& data,
                                std::size_t startRow,
                                const std::vector<double>& predictions) {
    double squaredError = 0.0;
    double percentageErrorSum = 0.0;
    double maxPercentageError = 0.0;

    for (std::size_t localRow = 0; localRow < predictions.size(); ++localRow) {
        const double actual = data.y[startRow + localRow];
        const double error = predictions[localRow] - actual;
        squaredError += error * error;

        // Preserve v1 behaviour: a zero target contributes zero percentage
        // error and remains in the denominator of the mean.
        if (actual != 0.0) {
            const double percentage = std::abs(error / actual) * 100.0;
            percentageErrorSum += percentage;
            maxPercentageError = std::max(maxPercentageError, percentage);
        }
    }

    Metrics metrics;
    metrics.cost = 0.5 * squaredError;
    metrics.percentageError = percentageErrorSum
                            / static_cast<double>(predictions.size());
    metrics.maxPercentageError = maxPercentageError;
    return metrics;
}

/**
 * Calculate gradients and diagnostics together.
 *
 * This is the central v2 optimisation. The old implementation expressed these
 * equations using transposes, generic matrix multiplication and Hadamard
 * products. V2 applies the same equations directly:
 *
 *   delta3_i   = (yHat_i - y_i) * yHat_i * (1 - yHat_i)
 *   gradW2_j  += hidden_ij * delta3_i
 *   delta2_ij  = delta3_i * W2_j * hidden_ij * (1 - hidden_ij)
 *   gradW1_kj += X_ik * delta2_ij
 *
 * Only one hidden-sized delta2 vector is needed for the current observation.
 */
inline Metrics calculateGradientsAndMetrics(const Dataset& data,
                                            std::size_t startRow,
                                            const Network& network,
                                            TrainingBuffers& buffers) {
    buffers.gradientW1.fill(0.0);
    std::fill(buffers.gradientW2.begin(), buffers.gradientW2.end(), 0.0);

    const std::size_t inputCount = network.w1.rows();
    const std::size_t hiddenCount = network.w1.cols();
    const std::size_t batchSize = buffers.hidden.rows();

    double squaredError = 0.0;
    double percentageErrorSum = 0.0;
    double maxPercentageError = 0.0;

    for (std::size_t localRow = 0; localRow < batchSize; ++localRow) {
        const std::size_t dataRow = startRow + localRow;
        const double actual = data.y[dataRow];
        const double prediction = buffers.predictions[localRow];
        const double error = prediction - actual;

        squaredError += error * error;
        if (actual != 0.0) {
            const double percentage = std::abs(error / actual) * 100.0;
            percentageErrorSum += percentage;
            maxPercentageError = std::max(maxPercentageError, percentage);
        }

        const double delta3 = error * prediction * (1.0 - prediction);
        const double* hiddenRow = buffers.hidden.rowData(localRow);

        for (std::size_t hidden = 0; hidden < hiddenCount; ++hidden) {
            const double activation = hiddenRow[hidden];
            buffers.gradientW2[hidden] += activation * delta3;
            buffers.delta2[hidden] = delta3 * network.w2[hidden]
                                   * activation * (1.0 - activation);
        }

        const double* xRow = data.x.rowData(dataRow);
        for (std::size_t input = 0; input < inputCount; ++input) {
            const double xValue = xRow[input];
            double* gradientRow = buffers.gradientW1.rowData(input);
            for (std::size_t hidden = 0; hidden < hiddenCount; ++hidden) {
                gradientRow[hidden] += xValue * buffers.delta2[hidden];
            }
        }
    }

    Metrics metrics;
    metrics.cost = 0.5 * squaredError;
    metrics.percentageError = percentageErrorSum / static_cast<double>(batchSize);
    metrics.maxPercentageError = maxPercentageError;
    return metrics;
}

/**
 * Apply the same classical momentum equation used by batchOnlineRun in v1:
 *
 *   velocity = momentum * velocity - learningRate * gradient
 *   weight   += velocity
 *
 * V1 performed this through separate scale/subtract/add matrix passes. V2
 * fuses those operations into one traversal of each weight array.
 */
inline void applyMomentumUpdate(Network& network,
                                const TrainingBuffers& buffers,
                                double learningRate,
                                double momentum) {
    std::vector<double>& weightsW1 = network.w1.values();
    std::vector<double>& velocityW1 = network.velocityW1.values();
    const std::vector<double>& gradientW1 = buffers.gradientW1.values();

    for (std::size_t index = 0; index < weightsW1.size(); ++index) {
        velocityW1[index] = momentum * velocityW1[index]
                          - learningRate * gradientW1[index];
        weightsW1[index] += velocityW1[index];
    }

    for (std::size_t hidden = 0; hidden < network.w2.size(); ++hidden) {
        network.velocityW2[hidden] = momentum * network.velocityW2[hidden]
                                   - learningRate * buffers.gradientW2[hidden];
        network.w2[hidden] += network.velocityW2[hidden];
    }
}

/** Print one compact training-progress line without forcing a stream flush. */
inline void printProgress(std::size_t batchIndex,
                          std::size_t updates,
                          const Metrics& metrics) {
    std::cout << "Batch " << batchIndex
              << " | updates = " << updates
              << " | cost = " << metrics.cost
              << " | percentage error = " << metrics.percentageError
              << " | max error = " << metrics.maxPercentageError
              << '\n';
}

}  // namespace simple_nn_v2

#endif  // SIMPLE_NN_V2_ENGINE_HPP
