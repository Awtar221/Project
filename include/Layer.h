#pragma once

#include "Activation.h"
#include "Matrix.h"

#include <iosfwd>
#include <memory>
#include <random>

namespace mlp {

// One fully-connected (dense) layer: output = activation(W * input + b).
// A Layer OWNS its weights, biases and activation function (composition).
class Layer {
public:
    Layer(int inputSize, int outputSize, std::unique_ptr<ActivationFunction> activation,
          std::mt19937& rng);
    Layer(Matrix weights, Matrix biases, std::unique_ptr<ActivationFunction> activation);

    // Forward pass. Remembers input and pre-activation z for backward().
    Matrix forward(const Matrix& input);

    // Backward pass. Takes dLoss/dOutput, updates W and b, returns dLoss/dInput
    // so the previous layer can continue the chain rule.
    Matrix backward(const Matrix& gradOutput, double learningRate);

    int inputSize() const { return weights_.cols(); }
    int outputSize() const { return weights_.rows(); }
    const ActivationFunction& activation() const { return *activation_; }

    void save(std::ostream& os) const;

private:
    Matrix weights_; // outputSize x inputSize
    Matrix biases_;  // outputSize x 1
    std::unique_ptr<ActivationFunction> activation_;

    Matrix lastInput_; // cached during forward()
    Matrix lastZ_;     // cached during forward()
};

} // namespace mlp
