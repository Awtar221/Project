# 4. Layer — `include/Layer.h`, `src/Layer.cpp`

## Purpose

One **fully-connected (dense)** layer: every input is connected to every neuron.

```
z      = W × input + b
output = activation(z)
```

## Data (private)

| Member | Shape | Meaning |
|---|---|---|
| `weights_` | outputSize × inputSize | `W` — learned |
| `biases_` | outputSize × 1 | `b` — learned |
| `activation_` | `unique_ptr<ActivationFunction>` | which non-linearity this layer uses |
| `lastInput_` | inputSize × 1 | remembered during `forward`, needed by `backward` |
| `lastZ_` | outputSize × 1 | remembered during `forward`, needed by `backward` |

**Composition:** a Layer *owns* its matrices and its activation. When the Layer is destroyed,
they are destroyed too. `unique_ptr` makes that ownership explicit and automatic (no `delete`).

Because `unique_ptr` cannot be copied, a `Layer` can be **moved but not copied** — that's
why `std::vector<Layer>` in `NeuralNetwork` uses `emplace_back` and moves.

## Constructors (overloaded)

1. `Layer(inputSize, outputSize, activation, rng)` — a **new** layer with random weights.
   The random range is `sqrt(6 / (in + out))` (Xavier initialisation) so the starting
   outputs are neither tiny nor huge. Biases start at 0.
2. `Layer(weights, biases, activation)` — a layer **rebuilt from a saved file**.
   Checks the bias shape matches the weights.

Both throw `invalid_argument` if `activation` is null.

## `forward(input)` — making a prediction

```cpp
lastInput_ = input;
lastZ_ = weights_ * input + biases_;
return lastZ_.apply([&act](double z) { return act.activate(z); });
```

Saves `input` and `z` because learning needs them later.

## `backward(gradOutput, learningRate)` — learning

`gradOutput` = how much the loss changes if this layer's output changes (∂L/∂output).

```cpp
delta     = gradOutput ⊙ activation'(lastZ_)       // chain rule through activation
gradInput = Wᵀ × delta                              // pass the blame to the previous layer
W        -= learningRate × (delta × lastInputᵀ)     // gradient descent on weights
b        -= learningRate × delta                    // gradient descent on biases
return gradInput;
```

`gradInput` is computed **before** the weights change — otherwise the previous layer
would receive a gradient based on the new weights, which is mathematically wrong.

## `save(os)`

Writes one block of the model file:

```
<inputSize> <outputSize> <activationName>
<weights, one row per line>
<biases on one line>
```

## Small getters

`inputSize()`, `outputSize()` (read from the weight shape — no separate variable that
could get out of sync), and `activation()` (returns a `const` reference — callers can read
the name but cannot change the activation).
