#include "run_modes.hpp"

#include <chrono>
#include <cstddef>
#include <ctime>
#include <exception>
#include <iostream>
#include <random>

/*
 * Simple Neural Network - v2
 * ==========================
 *
 * This is an optimised companion to the original 2017 main.cpp. The historical
 * file remains untouched at the repository root. V2 deliberately preserves the
 * same model mathematics and text-file formats while making the implementation
 * substantially more cache-friendly and avoiding large temporary matrices.
 */

int main() {
    using namespace simple_nn_v2;

    try {
        // ------------------------------------------------------------------
        // Run mode. These defaults match the checked-in v1 configuration.
        // Normally exactly one mode should be true.
        // ------------------------------------------------------------------
        const bool batchOnline = false;
        const bool offline = false;
        const bool test = true;

        // ------------------------------------------------------------------
        // Network setup. The historical architecture is intentionally kept:
        //
        //      11 inputs -> 16 sigmoid hidden nodes -> 1 sigmoid output
        //
        // V2 remains a faithful performance rewrite, so it does not add bias
        // parameters or change the activation/loss functions.
        // ------------------------------------------------------------------
        const std::size_t numberOfVariables = 11;
        const std::size_t nodesInFirstHiddenLayer = 16;
        const double rangeWone = 4.0;
        const double rangeWtwo = 4.0;
        const bool randomiseWeights = false;
        const double percentageErrorTarget = 3.9;
        const double learningRate = 0.0001;
        const double momentum = 0.75;

        // ------------------------------------------------------------------
        // Batch-online setup. `iterations` is the number of sequential chunks;
        // `exampleSize` is the number of observations in each chunk.
        // ------------------------------------------------------------------
        const std::size_t iterations = 1;
        const std::size_t exampleSize = 13853;
        std::size_t numberOfDescents = 1000000;
        const std::size_t times = 1;

        // Printing every descent was expensive in v1. This controls progress
        // frequency without changing how often the weights are updated.
        const std::size_t logEvery = 1000;

        // ------------------------------------------------------------------
        // Offline/test dimensions retained from v1.
        // ------------------------------------------------------------------
        const std::size_t offlineRows = 10000;
        const std::size_t testRows = 13853;

        // A time-based seed preserves v1's non-reproducible default behaviour.
        // Use a fixed integer here when repeatable experiments are preferable.
        std::mt19937 generator(static_cast<unsigned int>(std::time(nullptr)));

        const auto startTime = std::chrono::steady_clock::now();

        if (batchOnline) {
            for (std::size_t run = 0; run < times; ++run) {
                OptimiserConfig optimiser;
                optimiser.maxDescents = numberOfDescents;
                optimiser.logEvery = logEvery;
                optimiser.percentageErrorTarget = percentageErrorTarget;
                optimiser.learningRate = learningRate;
                optimiser.momentum = momentum;

                batchOnlineRun(numberOfVariables,
                               nodesInFirstHiddenLayer,
                               iterations,
                               exampleSize,
                               optimiser,
                               rangeWone,
                               rangeWtwo,
                               randomiseWeights,
                               generator);

                // Preserve the v1 behaviour of doubling the descent allowance
                // between repeated batch-online runs.
                numberOfDescents += numberOfDescents;
            }
        }

        if (offline) {
            offlineRun(numberOfVariables,
                       offlineRows,
                       nodesInFirstHiddenLayer,
                       numberOfDescents,
                       percentageErrorTarget,
                       rangeWone,
                       rangeWtwo,
                       randomiseWeights,
                       generator);
        }

        if (test) {
            testRun(numberOfVariables, testRows, nodesInFirstHiddenLayer);
        }

        const auto endTime = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = endTime - startTime;
        std::cout << "Elapsed time = " << elapsed.count() << " seconds\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
