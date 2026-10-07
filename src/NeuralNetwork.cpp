#include "NeuralNetwork.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace mlp {

NeuralNetwork::NeuralNetwork(int inputSize, unsigned seed) : inputSize_(inputSize), rng_(seed) {
    if (inputSize < 1) {
        throw std::invalid_argument("Network needs at least one input");
    }
}

void NeuralNetwork::addLayer(int neurons, const std::string& activationName) {
    if (neurons < 1) {
        throw std::invalid_argument("A layer needs at least one neuron");
    }
    // A new layer takes as many inputs as the previous layer has outputs.
    int in = layers_.empty() ? inputSize_ : layers_.back().outputSize();
    layers_.emplace_back(in, neurons, makeActivation(activationName), rng_);
}

int NeuralNetwork::outputSize() const {
    return layers_.empty() ? 0 : layers_.back().outputSize();
}

void NeuralNetwork::requireReady() const {
    if (layers_.empty()) {
        throw std::logic_error("Network has no layers yet");
    }
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    Matrix a = input;
    for (Layer& layer : layers_) {
        a = layer.forward(a);
    }
    return a;
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    requireReady();
    if (static_cast<int>(input.size()) != inputSize_) {
        throw std::invalid_argument("Expected " + std::to_string(inputSize_) + " inputs, got " +
                                    std::to_string(input.size()));
    }
    return forward(Matrix(input)).toVector();
}

double NeuralNetwork::train(Dataset data, int epochs, double learningRate, int reportEvery,
                            std::ostream& log) {
    requireReady();
    if (data.empty()) {
        throw std::invalid_argument("Cannot train on an empty dataset");
    }
    if (data.inputSize() != inputSize_ || data.outputSize() != outputSize()) {
        throw std::invalid_argument("Dataset shape does not match network shape");
    }
    if (epochs < 1 || learningRate <= 0.0) {
        throw std::invalid_argument("Epochs and learning rate must be positive");
    }

    double epochLoss = 0.0;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        data.shuffle(rng_); // data is a copy, so the caller's order is untouched
        epochLoss = 0.0;

        for (const Sample& s : data.samples()) {
            Matrix target(s.target);
            Matrix output = forward(Matrix(s.input));
            Matrix error = output - target;

            for (double e : error.toVector()) {
                epochLoss += e * e;
            }

            // Backpropagation: walk the layers from last to first.
            Matrix grad = error * (2.0 / target.rows()); // dMSE/dOutput
            for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                grad = it->backward(grad, learningRate);
            }
        }
        epochLoss /= static_cast<double>(data.size() * data.outputSize());

        if (reportEvery > 0 && (epoch % reportEvery == 0 || epoch == epochs)) {
            log << "Epoch " << std::setw(6) << epoch << "  loss = " << std::fixed
                << std::setprecision(6) << epochLoss << '\n';
        }
    }
    return epochLoss;
}

Evaluation NeuralNetwork::evaluate(const Dataset& data) {
    requireReady();
    if (data.empty()) {
        throw std::invalid_argument("Cannot evaluate on an empty dataset");
    }
    double loss = 0.0;
    int correct = 0;
    for (const Sample& s : data.samples()) {
        std::vector<double> out = predict(s.input);
        if (out.size() != s.target.size()) {
            throw std::invalid_argument("Dataset shape does not match network shape");
        }
        bool allMatch = true;
        for (size_t i = 0; i < out.size(); ++i) {
            double e = out[i] - s.target[i];
            loss += e * e;
            if (std::round(out[i]) != std::round(s.target[i])) allMatch = false;
        }
        if (allMatch) ++correct;
    }
    return {loss / static_cast<double>(data.size() * data.outputSize()),
            static_cast<double>(correct) / static_cast<double>(data.size())};
}

void NeuralNetwork::save(const std::string& filename) const {
    requireReady();
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Cannot write file: " + filename);
    }
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "MLP " << inputSize_ << ' ' << layers_.size() << '\n';
    for (const Layer& layer : layers_) {
        layer.save(out);
    }
}

NeuralNetwork NeuralNetwork::load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    std::string magic;
    int inputSize = 0;
    size_t layerCount = 0;
    if (!(in >> magic >> inputSize >> layerCount) || magic != "MLP") {
        throw std::runtime_error("Not a valid model file: " + filename);
    }

    NeuralNetwork net(inputSize);
    int expectedIn = inputSize;
    for (size_t l = 0; l < layerCount; ++l) {
        int in_ = 0, out_ = 0;
        std::string act;
        if (!(in >> in_ >> out_ >> act) || in_ != expectedIn || out_ < 1) {
            throw std::runtime_error("Corrupt layer header in " + filename);
        }
        Matrix w(out_, in_), b(out_, 1);
        for (int r = 0; r < out_; ++r)
            for (int c = 0; c < in_; ++c)
                in >> w(r, c);
        for (int r = 0; r < out_; ++r)
            in >> b(r, 0);
        if (!in) {
            throw std::runtime_error("Model file ended early: " + filename);
        }
        net.layers_.emplace_back(std::move(w), std::move(b), makeActivation(act));
        expectedIn = out_;
    }
    return net;
}

std::ostream& operator<<(std::ostream& os, const NeuralNetwork& net) {
    os << "Input layer : " << net.inputSize_ << " neuron(s)\n";
    for (size_t i = 0; i < net.layers_.size(); ++i) {
        const Layer& l = net.layers_[i];
        bool isOutput = (i + 1 == net.layers_.size());
        os << (isOutput ? "Output layer: " : "Hidden " + std::to_string(i + 1) + "    : ")
           << l.outputSize() << " neuron(s), " << l.activation().name() << ", "
           << (l.inputSize() * l.outputSize() + l.outputSize()) << " parameters\n";
    }
    if (net.layers_.empty()) os << "(no layers yet)\n";
    return os;
}

} // namespace mlp
