#pragma once

#include "Dataset.h"
#include "Layer.h"

#include <iosfwd>
#include <random>
#include <string>
#include <vector>

namespace mlp {

// Result of checking the network against a dataset.
struct Evaluation {
    double loss;     // mean squared error
    double accuracy; // fraction of samples where every rounded output matches the target
};

// The Multi-Layer Perceptron: an ordered stack of Layers (composition).
class NeuralNetwork {
public:
    explicit NeuralNetwork(int inputSize, unsigned seed = 42);

    void addLayer(int neurons, const std::string& activationName);

    std::vector<double> predict(const std::vector<double>& input);

    // Stochastic gradient descent with mean-squared-error loss.
    // Prints progress every `reportEvery` epochs (0 = silent). Returns final loss.
    double train(Dataset data, int epochs, double learningRate, int reportEvery, std::ostream& log);

    Evaluation evaluate(const Dataset& data);

    void save(const std::string& filename) const;
    static NeuralNetwork load(const std::string& filename);

    int inputSize() const { return inputSize_; }
    int outputSize() const;
    size_t layerCount() const { return layers_.size(); }

    friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork& net);

private:
    int inputSize_;
    std::mt19937 rng_;
    std::vector<Layer> layers_;

    Matrix forward(const Matrix& input);
    void requireReady() const;
};

} // namespace mlp
