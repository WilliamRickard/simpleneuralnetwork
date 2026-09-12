#ifndef SIMPLE_NN_V2_RUN_MODES_HPP
#define SIMPLE_NN_V2_RUN_MODES_HPP

#include "engine.hpp"
#include "io.hpp"

#include <cstddef>
#include <iostream>
#include <random>

namespace simple_nn_v2 {

/**
 * Fit one sequential batch until the target is met or maxDescents updates have
 * been made. Scratch buffers are allocated once, outside the descent loop.
 *
 * The original code intended numberOfDescents to stop the loop, but changed
 * `cost` while the loop condition depended on percentageError. V2 implements
 * the documented hard limit directly.
 */
inline TrainResult trainRange(const Dataset& data,
                              std::size_t startRow,
                              std::size_t batchSize,
                              std::size_t batchIndex,
                              Network& network,
                              const OptimiserConfig& config) {
    TrainingBuffers buffers(batchSize, network.w1.rows(), network.w1.cols());
    TrainResult result;

    while (true) {
        forwardRange(data, startRow, network, buffers);
        result.metrics = calculateGradientsAndMetrics(data, startRow, network, buffers);

        const bool targetReached =
            result.metrics.percentageError <= config.percentageErrorTarget;
        const bool updateLimitReached = result.updates >= config.maxDescents;
        const bool logNow = config.logEvery != 0
                         && result.updates % config.logEvery == 0;

        if (logNow || targetReached || updateLimitReached) {
            printProgress(batchIndex, result.updates, result.metrics);
        }
        if (targetReached || updateLimitReached) {
            result.reachedTarget = targetReached;
            break;
        }

        applyMomentumUpdate(network, buffers, config.learningRate, config.momentum);
        ++result.updates;
    }

    // Match v1's convention: ybar.txt contains the most recently processed
    // batch. The final forward pass already contains the required predictions.
    writeVector(buffers.predictions, "ybar.txt");
    return result;
}

/**
 * Optimised equivalent of the original batchOnlineRun().
 *
 * Data are loaded once. Each sequential batch is represented by a starting row
 * and size rather than copied into separate X/y matrices. Momentum persists
 * between batches, matching the lifetime of v1's deltawone/deltawtwo arrays.
 */
inline void batchOnlineRun(std::size_t inputCount,
                           std::size_t hiddenCount,
                           std::size_t iterations,
                           std::size_t exampleSize,
                           const OptimiserConfig& optimiser,
                           double rangeW1,
                           double rangeW2,
                           bool randomise,
                           std::mt19937& generator) {
    const std::size_t totalRows = iterations * exampleSize;
    Dataset data = loadDataset(totalRows, inputCount);
    Network network(inputCount, hiddenCount);

    if (randomise) {
        randomiseWeights(network, rangeW1, rangeW2, generator);
    } else {
        loadWeights(network);
    }

    for (std::size_t batch = 0; batch < iterations; ++batch) {
        const std::size_t startRow = batch * exampleSize;
        const TrainResult result = trainRange(
            data, startRow, exampleSize, batch, network, optimiser);

        // The README for v1 describes autosaving after the descent allowance,
        // so save useful progress whether the target or the hard limit ended
        // the batch.
        saveWeights(network);
        if (!result.reachedTarget) {
            std::cout << "Batch " << batch
                      << " reached the maximum number of descents before the target.\n";
        }
    }
}

/**
 * Optimised equivalent of testRun().
 *
 * V1 prints every matrix, which can dominate runtime for thousands of rows.
 * V2 prints compact diagnostics and keeps the detailed predictions in ybar.txt.
 */
inline void testRun(std::size_t inputCount,
                    std::size_t rowCount,
                    std::size_t hiddenCount) {
    Dataset data = loadDataset(rowCount, inputCount);
    Network network(inputCount, hiddenCount);
    loadWeights(network);

    TrainingBuffers buffers(rowCount, inputCount, hiddenCount);
    forwardRange(data, 0, network, buffers);
    const Metrics metrics = calculateMetrics(data, 0, buffers.predictions);

    std::cout << "Test cost = " << metrics.cost
              << " | percentage error = " << metrics.percentageError
              << " | max error = " << metrics.maxPercentageError
              << '\n';
    writeVector(buffers.predictions, "ybar.txt");
}

/**
 * Optimised equivalent of offlineRun().
 *
 * V1's offline update subtracts the full gradient directly. That is equivalent
 * to learningRate=1 and momentum=0, which v2 preserves. A finite descent limit
 * is added so an unreachable target cannot create an unbounded loop.
 */
inline void offlineRun(std::size_t inputCount,
                       std::size_t rowCount,
                       std::size_t hiddenCount,
                       std::size_t maxDescents,
                       double percentageErrorTarget,
                       double rangeW1,
                       double rangeW2,
                       bool randomise,
                       std::mt19937& generator) {
    Dataset data = loadDataset(rowCount, inputCount);
    Network network(inputCount, hiddenCount);

    if (randomise) {
        randomiseWeights(network, rangeW1, rangeW2, generator);
    } else {
        loadWeights(network);
    }

    OptimiserConfig optimiser;
    optimiser.maxDescents = maxDescents;
    optimiser.percentageErrorTarget = percentageErrorTarget;
    optimiser.learningRate = 1.0;
    optimiser.momentum = 0.0;

    const TrainResult result = trainRange(data, 0, rowCount, 0, network, optimiser);
    saveWeights(network);

    if (!result.reachedTarget) {
        std::cout << "Offline training reached the maximum number of descents "
                  << "before the target.\n";
    }
}

}  // namespace simple_nn_v2

#endif  // SIMPLE_NN_V2_RUN_MODES_HPP
