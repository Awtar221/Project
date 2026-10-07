#pragma once

#include <functional>
#include <iosfwd>
#include <random>
#include <vector>

namespace mlp {

// Small dense matrix of doubles, stored row-major in one vector.
// Every layer of the network is built from these: weights, biases, inputs, outputs.
class Matrix {
public:
    Matrix();
    Matrix(int rows, int cols, double fill = 0.0);
    explicit Matrix(const std::vector<double>& column); // column vector (n x 1)

    static Matrix random(int rows, int cols, double range, std::mt19937& rng);

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    double& operator()(int r, int c);
    double operator()(int r, int c) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const; // matrix product
    Matrix operator*(double scalar) const;       // overload: scale every element
    Matrix& operator-=(const Matrix& other);

    Matrix hadamard(const Matrix& other) const;  // element-wise product
    Matrix transpose() const;
    Matrix apply(const std::function<double(double)>& f) const;

    std::vector<double> toVector() const;

private:
    int rows_;
    int cols_;
    std::vector<double> data_;

    void requireSameShape(const Matrix& other, const char* op) const;
};

std::ostream& operator<<(std::ostream& os, const Matrix& m);

} // namespace mlp
