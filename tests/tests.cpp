// Self-check: compile with the src/ files (see README) and run. Prints PASS/FAIL per test.
#include "Activation.h"
#include "Dataset.h"
#include "Matrix.h"
#include "NeuralNetwork.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace mlp;

static int failures = 0;

static void check(const std::string& id, const std::string& desc, const std::function<bool()>& test) {
    bool ok = false;
    try {
        ok = test();
    } catch (const std::exception& e) {
        std::cout << "  (unexpected exception: " << e.what() << ")\n";
    }
    std::cout << (ok ? "PASS " : "FAIL ") << id << "  " << desc << '\n';
    if (!ok) ++failures;
}

template <typename E>
static bool throws(const std::function<void()>& f) {
    try {
        f();
    } catch (const E&) {
        return true;
    }
    return false;
}

static bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

int main() {
    check("T01", "Matrix product 2x2 * 2x1", [] {
        Matrix a(2, 2);
        a(0, 0) = 1; a(0, 1) = 2; a(1, 0) = 3; a(1, 1) = 4;
        Matrix r = a * Matrix(std::vector<double>{1, 1});
        return near(r(0, 0), 3) && near(r(1, 0), 7);
    });
    check("T02", "Matrix product with wrong shapes throws", [] {
        return throws<std::invalid_argument>([] { Matrix(2, 3) * Matrix(2, 3); });
    });
    check("T03", "Matrix transpose swaps shape", [] {
        Matrix t = Matrix(2, 3).transpose();
        return t.rows() == 3 && t.cols() == 2;
    });
    check("T04", "Matrix index out of range throws", [] {
        return throws<std::out_of_range>([] { Matrix(2, 2)(5, 0); });
    });
    check("T05", "Matrix * scalar and hadamard", [] {
        Matrix m(1, 2, 3.0);
        return near((m * 2.0)(0, 1), 6) && near(m.hadamard(m)(0, 0), 9);
    });
    check("T06", "Sigmoid(0) = 0.5, derivative 0.25", [] {
        Sigmoid s;
        return near(s.activate(0), 0.5) && near(s.derivative(0), 0.25);
    });
    check("T07", "ReLU clips negatives", [] {
        ReLU r;
        return near(r.activate(-3), 0) && near(r.activate(2), 2) && near(r.derivative(-1), 0);
    });
    check("T08", "Polymorphism via base pointer", [] {
        std::unique_ptr<ActivationFunction> a = makeActivation("tanh");
        return a->name() == "tanh" && near(a->activate(0), 0) && near(a->derivative(0), 1);
    });
    check("T09", "Unknown activation name throws", [] {
        return throws<std::invalid_argument>([] { makeActivation("banana"); });
    });
    check("T10", "Built-in XOR dataset has 4 samples, 2 in / 1 out", [] {
        Dataset d = Dataset::xorGate();
        return d.size() == 4 && d.inputSize() == 2 && d.outputSize() == 1;
    });
    check("T11", "Adding mismatched sample throws", [] {
        Dataset d = Dataset::xorGate();
        return throws<std::invalid_argument>([&d] { d.addSample({1, 2, 3}, {0}); });
    });
    check("T12", "Missing CSV file throws runtime_error", [] {
        return throws<std::runtime_error>([] { Dataset::fromCsv("no_such_file.csv", 1); });
    });
    check("T13", "Predict with no layers throws", [] {
        NeuralNetwork net(2);
        return throws<std::logic_error>([&net] { net.predict({0, 0}); });
    });
    check("T14", "Predict with wrong input count throws", [] {
        NeuralNetwork net(2);
        net.addLayer(1, "sigmoid");
        return throws<std::invalid_argument>([&net] { net.predict({1, 2, 3}); });
    });
    check("T15", "Network learns XOR to 100% accuracy", [] {
        NeuralNetwork net(2);
        net.addLayer(4, "tanh");
        net.addLayer(1, "sigmoid");
        std::ostringstream quiet;
        net.train(Dataset::xorGate(), 5000, 0.5, 0, quiet);
        Evaluation ev = net.evaluate(Dataset::xorGate());
        return ev.accuracy == 1.0 && ev.loss < 0.05;
    });
    check("T16", "Save then load gives identical predictions", [] {
        NeuralNetwork net(2);
        net.addLayer(3, "relu");
        net.addLayer(1, "sigmoid");
        const char* file = "test_model.tmp";
        net.save(file);
        NeuralNetwork loaded = NeuralNetwork::load(file);
        std::remove(file);
        return near(net.predict({0.3, 0.7})[0], loaded.predict({0.3, 0.7})[0], 1e-12);
    });
    check("T17", "Loading a non-model file throws", [] {
        return throws<std::runtime_error>([] { NeuralNetwork::load("data/xor.csv"); });
    });

    std::cout << (failures == 0 ? "\nAll tests passed.\n" : "\nSome tests FAILED.\n");
    return failures == 0 ? 0 : 1;
}
