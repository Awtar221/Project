#include "Activation.h"

#include <cmath>
#include <stdexcept>

namespace mlp {

double Sigmoid::activate(double z) const {
    return 1.0 / (1.0 + std::exp(-z));
}

double Sigmoid::derivative(double z) const {
    double s = activate(z);
    return s * (1.0 - s);
}

double ReLU::activate(double z) const {
    return z > 0.0 ? z : 0.0;
}

double ReLU::derivative(double z) const {
    return z > 0.0 ? 1.0 : 0.0;
}

double Tanh::activate(double z) const {
    return std::tanh(z);
}

double Tanh::derivative(double z) const {
    double t = std::tanh(z);
    return 1.0 - t * t;
}

std::unique_ptr<ActivationFunction> makeActivation(const std::string& name) {
    if (name == "sigmoid") return std::make_unique<Sigmoid>();
    if (name == "relu") return std::make_unique<ReLU>();
    if (name == "tanh") return std::make_unique<Tanh>();
    throw std::invalid_argument("Unknown activation function: " + name);
}

} // namespace mlp
