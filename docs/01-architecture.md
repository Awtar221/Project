# 1. Architecture Overview

## What is an MLP?

A **Multi-Layer Perceptron** is the simplest kind of neural network. It is a stack of
*layers*. Each layer takes a list of numbers in, multiplies them by learned **weights**,
adds a **bias**, and squashes the result with an **activation function**:

```
output = activation( W × input + b )
```

The output of one layer is the input of the next. **Training** means adjusting every
`W` and `b` so the final output gets closer to the target. This is done with
**backpropagation** (chain rule, computed from the last layer back to the first) and
**gradient descent** (nudge each weight a little in the direction that lowers the error).

The classic test is **XOR** — a single layer cannot learn it, but one hidden layer can.

## Classes and who owns what

```
App ──owns──► Dataset ──owns──► Sample (many)
 │
 └──owns──► NeuralNetwork ──owns──► Layer (many) ──owns──► Matrix (weights, biases, caches)
                                         │
                                         └──owns──► ActivationFunction  (abstract)
                                                        ▲      ▲      ▲
                                                    Sigmoid  ReLU   Tanh
```

| Class | File | One-line responsibility |
|---|---|---|
| `App` | `include/App.h`, `src/App.cpp` | Console menu: reads input, calls the right object, shows results |
| `NeuralNetwork` | `include/NeuralNetwork.h`, `src/NeuralNetwork.cpp` | Holds the layers; predict, train, evaluate, save, load |
| `Layer` | `include/Layer.h`, `src/Layer.cpp` | One dense layer: forward pass and backward (learning) pass |
| `ActivationFunction` | `include/Activation.h`, `src/Activation.cpp` | Abstract interface: `activate`, `derivative`, `name` |
| `Sigmoid`, `ReLU`, `Tanh` | same as above | Concrete activation functions |
| `Matrix` | `include/Matrix.h`, `src/Matrix.cpp` | Maths: add, subtract, multiply, transpose, apply a function |
| `Dataset` (+ `Sample`) | `include/Dataset.h`, `src/Dataset.cpp` | Training data: built-in gates, CSV loading, shuffling |
| `Evaluation` (struct) | `include/NeuralNetwork.h` | Result of evaluation: loss + accuracy |

Everything lives in the namespace `mlp` so names like `Matrix` or `Layer` cannot clash
with other code.

## Flow of a prediction (option 5)

1. `App::predictInput()` reads one number per input neuron.
2. Calls `NeuralNetwork::predict(vector<double>)`.
3. `predict` checks the input size, wraps it in a column `Matrix`, calls `forward`.
4. `NeuralNetwork::forward` loops over the layers: `a = layer.forward(a)`.
5. Each `Layer::forward` computes `W*a + b`, then calls `activation_->activate(z)` on every
   element — **polymorphism** picks Sigmoid / ReLU / Tanh at run time.
6. The final `Matrix` is turned back into a `vector<double>` and printed.

## Flow of training (option 3)

1. `App::trainNetwork()` reads epochs and learning rate.
2. `NeuralNetwork::train` repeats for each epoch:
   1. Shuffle the samples.
   2. For each sample: forward pass → error = output − target.
   3. Start gradient = `2 × error / n` (derivative of mean squared error).
   4. Walk layers **in reverse** calling `Layer::backward`, which updates that
      layer's weights and returns the gradient for the layer before it.
3. Every 10% of epochs it prints the loss.

## Why this design?

- **Each class does one job.** Matrix knows nothing about neural networks; Layer knows
  nothing about menus; App knows nothing about maths.
- **Adding a new activation function** (e.g. LeakyReLU) only needs a new subclass plus one
  line in `makeActivation` — no change to `Layer` or `NeuralNetwork`.
- **Composition, not inheritance, between Network/Layer/Matrix**: a network *has* layers,
  it *is not* a layer. Inheritance is only used where there is a true "is-a":
  Sigmoid *is an* activation function.
