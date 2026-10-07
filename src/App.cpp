#include "App.h"

#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace mlp {

App::App() : dataset_(Dataset::xorGate()) {}

void App::run() {
    std::cout << "=============================================\n"
              << "   Simple MLP (Multi-Layer Perceptron) in C++\n"
              << "=============================================\n"
              << "Dataset '" << dataset_.name() << "' loaded by default.\n";

    int choice = -1;
    do {
        showMenu();
        choice = readNumber<int>("Choose an option: ", 0, 8);
        // Every action is wrapped so an error returns to the menu instead of crashing.
        try {
            switch (choice) {
                case 1: loadDataset(); break;
                case 2: buildNetwork(); break;
                case 3: trainNetwork(); break;
                case 4: evaluateNetwork(); break;
                case 5: predictInput(); break;
                case 6: showSummary(); break;
                case 7: saveModel(); break;
                case 8: loadModel(); break;
                case 0: std::cout << "Goodbye!\n"; break;
            }
        } catch (const std::exception& e) {
            std::cout << "  Error: " << e.what() << '\n';
        }
    } while (choice != 0);
}

void App::showMenu() const {
    std::cout << "\n--------------- MAIN MENU ---------------\n"
              << " Dataset: " << dataset_.name() << " (" << dataset_.size() << " samples)"
              << "   Network: " << (network_ ? "ready" : "not built") << '\n'
              << " 1. Load dataset\n"
              << " 2. Build network\n"
              << " 3. Train network\n"
              << " 4. Evaluate network\n"
              << " 5. Predict a custom input\n"
              << " 6. Show network summary\n"
              << " 7. Save model to file\n"
              << " 8. Load model from file\n"
              << " 0. Exit\n"
              << "-----------------------------------------\n";
}

bool App::hasDataset() const {
    if (dataset_.empty()) {
        std::cout << "  No dataset loaded. Use option 1 first.\n";
        return false;
    }
    return true;
}

bool App::hasNetwork() const {
    if (!network_) {
        std::cout << "  No network yet. Use option 2 (build) or 8 (load) first.\n";
        return false;
    }
    return true;
}

void App::loadDataset() {
    std::cout << " 1. XOR gate\n 2. AND gate\n 3. OR gate\n 4. CSV file\n";
    switch (readNumber<int>("Dataset: ", 1, 4)) {
        case 1: dataset_ = Dataset::xorGate(); break;
        case 2: dataset_ = Dataset::andGate(); break;
        case 3: dataset_ = Dataset::orGate(); break;
        case 4: {
            std::string file = readLine("CSV path (e.g. data/xor.csv): ");
            int outputs = readNumber<int>("How many output columns (at the end of each row)? ", 1, 100);
            dataset_ = Dataset::fromCsv(file, outputs);
            break;
        }
    }
    std::cout << "  Loaded '" << dataset_.name() << "': " << dataset_.size() << " samples, "
              << dataset_.inputSize() << " input(s), " << dataset_.outputSize() << " output(s).\n";
    if (network_ && (network_->inputSize() != dataset_.inputSize() ||
                     network_->outputSize() != dataset_.outputSize())) {
        std::cout << "  Note: current network shape does not fit this dataset. Rebuild it (option 2).\n";
    }
}

void App::buildNetwork() {
    if (!hasDataset()) return;

    const std::vector<std::string> names = {"sigmoid", "relu", "tanh"};
    auto chooseActivation = [&names](const std::string& what) {
        std::cout << "  Activation for " << what << ": 1. sigmoid  2. relu  3. tanh\n";
        return names[readNumber<int>("  Choice: ", 1, 3) - 1];
    };

    auto net = std::make_unique<NeuralNetwork>(dataset_.inputSize());
    int hidden = readNumber<int>("Number of hidden layers (1-5): ", 1, 5);
    for (int i = 1; i <= hidden; ++i) {
        int neurons = readNumber<int>("Neurons in hidden layer " + std::to_string(i) + " (1-64): ", 1, 64);
        net->addLayer(neurons, chooseActivation("hidden layer " + std::to_string(i)));
    }
    // Output layer size is fixed by the dataset.
    net->addLayer(dataset_.outputSize(), chooseActivation("output layer"));

    network_ = std::move(net);
    std::cout << "  Network built:\n" << *network_;
}

void App::trainNetwork() {
    if (!hasDataset() || !hasNetwork()) return;
    int epochs = readNumber<int>("Epochs (1-100000): ", 1, 100000);
    double lr = readNumber<double>("Learning rate (0.0001-10): ", 0.0001, 10.0);
    int every = std::max(1, epochs / 10);
    double loss = network_->train(dataset_, epochs, lr, every, std::cout);
    std::cout << "  Training finished. Final loss: " << loss << '\n';
}

void App::evaluateNetwork() {
    if (!hasDataset() || !hasNetwork()) return;
    std::cout << std::fixed << std::setprecision(4);
    for (const Sample& s : dataset_.samples()) {
        std::vector<double> out = network_->predict(s.input);
        std::cout << "  input:";
        for (double v : s.input) std::cout << ' ' << v;
        std::cout << "  -> output:";
        for (double v : out) std::cout << ' ' << v;
        std::cout << "  (target:";
        for (double v : s.target) std::cout << ' ' << v;
        std::cout << ")\n";
    }
    Evaluation ev = network_->evaluate(dataset_);
    std::cout << "  MSE loss: " << ev.loss << "   Accuracy: " << std::setprecision(1)
              << ev.accuracy * 100.0 << "%\n";
}

void App::predictInput() {
    if (!hasNetwork()) return;
    std::vector<double> input;
    for (int i = 1; i <= network_->inputSize(); ++i) {
        input.push_back(readNumber<double>("Input " + std::to_string(i) + ": ", -1e6, 1e6));
    }
    std::cout << "  Output:";
    for (double v : network_->predict(input)) std::cout << ' ' << std::fixed << std::setprecision(4) << v;
    std::cout << '\n';
}

void App::showSummary() const {
    if (!hasNetwork()) return;
    std::cout << *network_;
}

void App::saveModel() const {
    if (!hasNetwork()) return;
    std::string file = readLine("Save to (e.g. data/model.txt): ");
    network_->save(file);
    std::cout << "  Model saved to " << file << '\n';
}

void App::loadModel() {
    std::string file = readLine("Load from (e.g. data/model.txt): ");
    network_ = std::make_unique<NeuralNetwork>(NeuralNetwork::load(file));
    std::cout << "  Model loaded:\n" << *network_;
}

std::string App::readLine(const std::string& prompt) {
    while (true) {
        std::cout << prompt;
        std::string line;
        if (!std::getline(std::cin, line)) throw std::runtime_error("Input closed");
        if (!line.empty()) return line;
        std::cout << "  Please enter something.\n";
    }
}

} // namespace mlp
