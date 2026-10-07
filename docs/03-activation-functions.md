# 3. Activation Functions — `include/Activation.h`, `src/Activation.cpp`

## Purpose

Without a non-linear activation function, stacking layers is pointless — the whole
network would collapse into one straight-line function and could never learn XOR.

## The abstract base class (abstraction)

```cpp
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;
    virtual double activate(double z) const = 0;
    virtual double derivative(double z) const = 0;
    virtual std::string name() const = 0;
};
```

- `= 0` makes these **pure virtual** → the class is **abstract**: you cannot write
  `ActivationFunction a;`. It only describes *what* an activation must do, not *how*.
- `derivative` is required because backpropagation needs the slope of the function.
- `name` is used for printing the summary and for saving the model to file.
- The **virtual destructor** guarantees the correct derived destructor runs when a
  `unique_ptr<ActivationFunction>` deletes a `Sigmoid`/`ReLU`/`Tanh`.

## The three derived classes (inheritance)

| Class | `activate(z)` | `derivative(z)` | Output range |
|---|---|---|---|
| `Sigmoid` | `1 / (1 + e^-z)` | `s(z) × (1 − s(z))` | 0 … 1 (good for yes/no outputs) |
| `ReLU` | `max(0, z)` | `1` if z > 0 else `0` | 0 … ∞ (fast, common in hidden layers) |
| `Tanh` | `tanh(z)` | `1 − tanh²(z)` | −1 … 1 (good hidden layer for XOR) |

Each is a genuine **"is-a"** relationship: Sigmoid *is an* activation function. Each uses
`override` so the compiler checks the signature matches the base class.

## Polymorphism in action

In `Layer::forward` (`src/Layer.cpp`):

```cpp
const ActivationFunction& act = *activation_;
return lastZ_.apply([&act](double z) { return act.activate(z); });
```

`act` is a **base-class reference**. Which `activate` actually runs (Sigmoid's, ReLU's or
Tanh's) is decided **at run time** by the real object — that is dynamic polymorphism.
`Layer` never needs an `if (type == "relu")`.

## The factory function (design pattern)

```cpp
std::unique_ptr<ActivationFunction> makeActivation(const std::string& name);
```

Turns the text `"sigmoid"`, `"relu"` or `"tanh"` into the right object, or throws
`invalid_argument` for an unknown name. Used in two places:

- `NeuralNetwork::addLayer` — when the user builds a network from the menu
- `NeuralNetwork::load` — when reading the activation name back from a model file

This is the **Factory** pattern: callers ask for "a relu" without knowing the class.

## How to add a new activation (e.g. LeakyReLU)

1. Add `class LeakyReLU : public ActivationFunction { ... };` in `Activation.h`.
2. Implement `activate`/`derivative` in `Activation.cpp`.
3. Add one line in `makeActivation`: `if (name == "leakyrelu") return std::make_unique<LeakyReLU>();`
4. Add it to the `names` list in `App::buildNetwork`.

No change is needed in `Layer` or `NeuralNetwork` — that's the payoff of polymorphism.
