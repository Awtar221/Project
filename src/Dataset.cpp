#include "Dataset.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mlp {

Dataset::Dataset(std::string name) : name_(std::move(name)) {}

Dataset Dataset::xorGate() {
    Dataset d("XOR");
    d.addSample({0, 0}, {0});
    d.addSample({0, 1}, {1});
    d.addSample({1, 0}, {1});
    d.addSample({1, 1}, {0});
    return d;
}

Dataset Dataset::andGate() {
    Dataset d("AND");
    d.addSample({0, 0}, {0});
    d.addSample({0, 1}, {0});
    d.addSample({1, 0}, {0});
    d.addSample({1, 1}, {1});
    return d;
}

Dataset Dataset::orGate() {
    Dataset d("OR");
    d.addSample({0, 0}, {0});
    d.addSample({0, 1}, {1});
    d.addSample({1, 0}, {1});
    d.addSample({1, 1}, {1});
    return d;
}

Dataset Dataset::fromCsv(const std::string& filename, int outputCount) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    if (outputCount < 1) {
        throw std::invalid_argument("Output count must be at least 1");
    }

    Dataset d(filename);
    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        if (line.empty() || line[0] == '#') continue; // skip blanks and comments

        std::vector<double> values;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try {
                values.push_back(std::stod(cell));
            } catch (const std::exception&) {
                throw std::runtime_error("Line " + std::to_string(lineNo) + ": '" + cell +
                                         "' is not a number");
            }
        }
        if (static_cast<int>(values.size()) <= outputCount) {
            throw std::runtime_error("Line " + std::to_string(lineNo) +
                                     ": needs more columns than the output count");
        }
        auto split = values.end() - outputCount;
        d.addSample(std::vector<double>(values.begin(), split), std::vector<double>(split, values.end()));
    }
    if (d.empty()) {
        throw std::runtime_error("File has no data rows: " + filename);
    }
    return d;
}

void Dataset::addSample(const std::vector<double>& input, const std::vector<double>& target) {
    addSample(Sample{input, target});
}

void Dataset::addSample(const Sample& sample) {
    if (sample.input.empty() || sample.target.empty()) {
        throw std::invalid_argument("Sample input and target cannot be empty");
    }
    if (!samples_.empty() && (static_cast<int>(sample.input.size()) != inputSize() ||
                              static_cast<int>(sample.target.size()) != outputSize())) {
        throw std::invalid_argument("Sample size does not match the rest of the dataset");
    }
    samples_.push_back(sample);
}

void Dataset::shuffle(std::mt19937& rng) {
    std::shuffle(samples_.begin(), samples_.end(), rng);
}

int Dataset::inputSize() const {
    return samples_.empty() ? 0 : static_cast<int>(samples_.front().input.size());
}

int Dataset::outputSize() const {
    return samples_.empty() ? 0 : static_cast<int>(samples_.front().target.size());
}

} // namespace mlp
