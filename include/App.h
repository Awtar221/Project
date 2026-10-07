#pragma once

#include "Dataset.h"
#include "NeuralNetwork.h"

#include <iostream>
#include <limits>
#include <memory>
#include <string>

namespace mlp {

// Console menu. Owns the current dataset and network and routes user choices to them.
class App {
public:
    App();
    void run();

private:
    Dataset dataset_;
    std::unique_ptr<NeuralNetwork> network_; // empty until the user builds or loads one

    void showMenu() const;
    void loadDataset();
    void buildNetwork();
    void trainNetwork();
    void evaluateNetwork();
    void predictInput();
    void showSummary() const;
    void saveModel() const;
    void loadModel();

    bool hasDataset() const;
    bool hasNetwork() const;

    // Template: read any numeric type, re-asking until it is valid and within [min, max].
    template <typename T>
    static T readNumber(const std::string& prompt, T min, T max) {
        while (true) {
            std::cout << prompt;
            T value;
            if (std::cin >> value && value >= min && value <= max) {
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                return value;
            }
            if (std::cin.eof()) throw std::runtime_error("Input closed");
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "  Invalid input. Enter a value between " << min << " and " << max << ".\n";
        }
    }

    static std::string readLine(const std::string& prompt);
};

} // namespace mlp
