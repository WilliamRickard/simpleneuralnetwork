#ifndef SIMPLE_NN_V2_IO_HPP
#define SIMPLE_NN_V2_IO_HPP

#include "model.hpp"

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace simple_nn_v2 {

/** Read an exact-size matrix from a whitespace-delimited text file. */
inline Matrix readMatrix(const std::string& path,
                         std::size_t rows,
                         std::size_t cols) {
    std::ifstream input(path.c_str());
    if (!input) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    Matrix matrix(rows, cols);
    for (std::size_t index = 0; index < matrix.size(); ++index) {
        if (!(input >> matrix.values()[index])) {
            throw std::runtime_error("Insufficient or invalid data in file: " + path);
        }
    }
    return matrix;
}

/** Read an exact number of scalar values. */
inline std::vector<double> readVector(const std::string& path,
                                      std::size_t count) {
    std::ifstream input(path.c_str());
    if (!input) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::vector<double> values(count, 0.0);
    for (std::size_t index = 0; index < count; ++index) {
        if (!(input >> values[index])) {
            throw std::runtime_error("Insufficient or invalid data in file: " + path);
        }
    }
    return values;
}

/** Write the same whitespace-delimited matrix layout used by v1. */
inline void writeMatrix(const Matrix& matrix, const std::string& path) {
    std::ofstream output(path.c_str());
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    output << std::setprecision(10);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        const double* rowValues = matrix.rowData(row);
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            if (col != 0) {
                output << ' ';
            }
            output << rowValues[col];
        }
        // '\n' does not force a stream flush; std::endl would.
        output << '\n';
    }
}

/** Write one scalar per line, matching v1's W2 and ybar files. */
inline void writeVector(const std::vector<double>& values,
                        const std::string& path) {
    std::ofstream output(path.c_str());
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    output << std::setprecision(10);
    for (double value : values) {
        output << value << '\n';
    }
}

/** Load the historical InputVariables/OutputVariables file pair. */
inline Dataset loadDataset(std::size_t rows, std::size_t inputCount) {
    Dataset data;
    data.x = readMatrix("InputVariables.txt", rows, inputCount);
    data.y = readVector("OutputVariables.txt", rows);
    return data;
}

/** Load weight files with the same dimensions and layout as v1. */
inline void loadWeights(Network& network) {
    network.w1 = readMatrix("wone.txt", network.w1.rows(), network.w1.cols());
    network.w2 = readVector("wtwo.txt", network.w2.size());
}

/** Save weight files that can also be read by the original program. */
inline void saveWeights(const Network& network) {
    writeMatrix(network.w1, "wone.txt");
    writeVector(network.w2, "wtwo.txt");
}

/**
 * Initialise weights uniformly within the same configurable ranges as v1.
 *
 * Initialisation is not a hot path. std::mt19937 and uniform_real_distribution
 * mainly make the intent clearer than repeated rand()/modulo operations.
 */
inline void randomiseWeights(Network& network,
                             double rangeW1,
                             double rangeW2,
                             std::mt19937& generator) {
    std::uniform_real_distribution<double> distributionW1(-rangeW1, rangeW1);
    std::uniform_real_distribution<double> distributionW2(-rangeW2, rangeW2);

    for (double& weight : network.w1.values()) {
        weight = distributionW1(generator);
    }
    for (double& weight : network.w2) {
        weight = distributionW2(generator);
    }
}

}  // namespace simple_nn_v2

#endif  // SIMPLE_NN_V2_IO_HPP
