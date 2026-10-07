#include "Matrix.h"

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

namespace mlp {

Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(int rows, int cols, double fill) : rows_(rows), cols_(cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    data_.assign(static_cast<size_t>(rows) * cols, fill);
}

Matrix::Matrix(const std::vector<double>& column)
    : rows_(static_cast<int>(column.size())), cols_(1), data_(column) {
    if (column.empty()) {
        throw std::invalid_argument("Cannot build a Matrix from an empty vector");
    }
}

Matrix Matrix::random(int rows, int cols, double range, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-range, range);
    Matrix m(rows, cols);
    for (double& v : m.data_) {
        v = dist(rng);
    }
    return m;
}

double& Matrix::operator()(int r, int c) {
    if (r < 0 || r >= rows_ || c < 0 || c >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[static_cast<size_t>(r) * cols_ + c];
}

double Matrix::operator()(int r, int c) const {
    if (r < 0 || r >= rows_ || c < 0 || c >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[static_cast<size_t>(r) * cols_ + c];
}

void Matrix::requireSameShape(const Matrix& other, const char* op) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(std::string("Shape mismatch in ") + op);
    }
}

Matrix Matrix::operator+(const Matrix& other) const {
    requireSameShape(other, "+");
    Matrix result(*this);
    std::transform(data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(),
                   [](double a, double b) { return a + b; });
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    requireSameShape(other, "-");
    Matrix result(*this);
    std::transform(data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(),
                   [](double a, double b) { return a - b; });
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Shape mismatch in matrix product");
    }
    Matrix result(rows_, other.cols_);
    for (int i = 0; i < rows_; ++i) {
        for (int k = 0; k < cols_; ++k) {
            double a = (*this)(i, k);
            for (int j = 0; j < other.cols_; ++j) {
                result(i, j) += a * other(k, j);
            }
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    return apply([scalar](double v) { return v * scalar; });
}

Matrix& Matrix::operator-=(const Matrix& other) {
    requireSameShape(other, "-=");
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Matrix Matrix::hadamard(const Matrix& other) const {
    requireSameShape(other, "hadamard");
    Matrix result(*this);
    std::transform(data_.begin(), data_.end(), other.data_.begin(), result.data_.begin(),
                   [](double a, double b) { return a * b; });
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

Matrix Matrix::apply(const std::function<double(double)>& f) const {
    Matrix result(*this);
    std::transform(data_.begin(), data_.end(), result.data_.begin(), f);
    return result;
}

std::vector<double> Matrix::toVector() const {
    return data_;
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    for (int i = 0; i < m.rows(); ++i) {
        os << "[ ";
        for (int j = 0; j < m.cols(); ++j) {
            os << std::fixed << std::setprecision(4) << m(i, j) << ' ';
        }
        os << "]\n";
    }
    return os;
}

} // namespace mlp
