#ifndef SIMPLE_NN_V2_MATRIX_HPP
#define SIMPLE_NN_V2_MATRIX_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

namespace simple_nn_v2 {

/**
 * Compact row-major matrix used throughout v2.
 *
 * The 2017 implementation used vector<vector<double>>, which performs one
 * allocation per row. Here all values live in one contiguous vector:
 *
 *     [ row 0 | row 1 | row 2 | ... ]
 *
 * This improves cache locality and removes thousands of small allocations for
 * large training sets. Bounds checks are deliberately omitted in the hot path;
 * matrix dimensions are established when each object is created.
 */
class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}

    Matrix(std::size_t rows, std::size_t cols, double initialValue = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, initialValue) {}

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return data_.size(); }

    double& operator()(std::size_t row, std::size_t col) noexcept {
        return data_[row * cols_ + col];
    }

    const double& operator()(std::size_t row, std::size_t col) const noexcept {
        return data_[row * cols_ + col];
    }

    double* rowData(std::size_t row) noexcept {
        return data_.data() + row * cols_;
    }

    const double* rowData(std::size_t row) const noexcept {
        return data_.data() + row * cols_;
    }

    std::vector<double>& values() noexcept { return data_; }
    const std::vector<double>& values() const noexcept { return data_; }

    void fill(double value) {
        std::fill(data_.begin(), data_.end(), value);
    }

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;
};

}  // namespace simple_nn_v2

#endif  // SIMPLE_NN_V2_MATRIX_HPP
