#include "Layer.h"

#include <cmath>
#include <ostream>
#include <stdexcept>

namespace mlp {

Layer::Layer(int inputSize, int outputSize, std::unique_ptr<ActivationFunction> activation,
             std::mt19937& rng)
    // Xavier-style range keeps starting weights small relative to the layer size.
    : weights_(Matrix::random(outputSize, inputSize, std::sqrt(6.0 / (inputSize + outputSize)), rng)),
      biases_(outputSize, 1, 0.0),
      activation_(std::move(activation)) {
    if (!activation_) {
        throw std::invalid_argument("Layer needs an activation function");
    }
}

Layer::Layer(Matrix weights, Matrix biases, std::unique_ptr<ActivationFunction> activation)
    : weights_(std::move(weights)), biases_(std::move(biases)), activation_(std::move(activation)) {
    if (!activation_) {
        throw std::invalid_argument("Layer needs an activation function");
    }
    if (biases_.rows() != weights_.rows() || biases_.cols() != 1) {
        throw std::invalid_argument("Bias shape does not match weights");
    }
}

Matrix Layer::forward(const Matrix& input) {
    lastInput_ = input;
    lastZ_ = weights_ * input + biases_;
    // Polymorphism: activation_ may be Sigmoid, ReLU or Tanh; the right one runs.
    const ActivationFunction& act = *activation_;
    return lastZ_.apply([&act](double z) { return act.activate(z); });
}

Matrix Layer::backward(const Matrix& gradOutput, double learningRate) {
    const ActivationFunction& act = *activation_;
    Matrix delta = gradOutput.hadamard(lastZ_.apply([&act](double z) { return act.derivative(z); }));

    Matrix gradInput = weights_.transpose() * delta; // computed BEFORE weights change

    weights_ -= (delta * lastInput_.transpose()) * learningRate;
    biases_ -= delta * learningRate;
    return gradInput;
}

void Layer::save(std::ostream& os) const {
    os << inputSize() << ' ' << outputSize() << ' ' << activation_->name() << '\n';
    for (int r = 0; r < weights_.rows(); ++r) {
        for (int c = 0; c < weights_.cols(); ++c) {
            os << weights_(r, c) << ' ';
        }
        os << '\n';
    }
    for (int r = 0; r < biases_.rows(); ++r) {
        os << biases_(r, 0) << ' ';
    }
    os << '\n';
}

} // namespace mlp
