#pragma once

#include <random>
#include <string>
#include <vector>

namespace mlp {

// One training example: input features and the expected output.
struct Sample {
    std::vector<double> input;
    std::vector<double> target;
};

// A named collection of samples. Can come from a built-in logic gate or a CSV file.
class Dataset {
public:
    Dataset() = default;
    explicit Dataset(std::string name);

    // Built-in datasets (2 inputs -> 1 output).
    static Dataset xorGate();
    static Dataset andGate();
    static Dataset orGate();

    // CSV: each row = inputs..., targets... ; the last `outputCount` columns are targets.
    static Dataset fromCsv(const std::string& filename, int outputCount);

    void addSample(const std::vector<double>& input, const std::vector<double>& target);
    void addSample(const Sample& sample); // overload: add an already-built Sample

    void shuffle(std::mt19937& rng);

    const std::string& name() const { return name_; }
    const std::vector<Sample>& samples() const { return samples_; }
    size_t size() const { return samples_.size(); }
    bool empty() const { return samples_.empty(); }
    int inputSize() const;
    int outputSize() const;

private:
    std::string name_;
    std::vector<Sample> samples_;
};

} // namespace mlp
