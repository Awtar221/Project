#pragma once

#include <memory>
#include <string>

namespace mlp {

// Abstract base class: every activation function must say how to activate a value
// and how to compute its derivative (needed for backpropagation).
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;

    virtual double activate(double z) const = 0;
    virtual double derivative(double z) const = 0; // d(activate)/dz
    virtual std::string name() const = 0;
};

class Sigmoid : public ActivationFunction {
public:
    double activate(double z) const override;
    double derivative(double z) const override;
    std::string name() const override { return "sigmoid"; }
};

class ReLU : public ActivationFunction {
public:
    double activate(double z) const override;
    double derivative(double z) const override;
    std::string name() const override { return "relu"; }
};

class Tanh : public ActivationFunction {
public:
    double activate(double z) const override;
    double derivative(double z) const override;
    std::string name() const override { return "tanh"; }
};

// Factory: turns a name ("sigmoid", "relu", "tanh") into the right object.
// Used by the menu and when loading a saved model from file.
std::unique_ptr<ActivationFunction> makeActivation(const std::string& name);

} // namespace mlp
