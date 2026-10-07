# 5. NeuralNetwork — `include/NeuralNetwork.h`, `src/NeuralNetwork.cpp`

## Purpose

The MLP itself: an ordered list of `Layer`s plus the operations a user cares about —
add a layer, predict, train, evaluate, save, load.

## Data (private)

| Member | Meaning |
|---|---|
| `inputSize_` | How many numbers go into the first layer |
| `rng_` | `std::mt19937` random generator, seeded (default 42) → same results every run, easy to demo/test |
| `layers_` | `std::vector<Layer>` — the layers in order (**composition**) |

## Methods

### `NeuralNetwork(int inputSize, unsigned seed = 42)`
Constructor. Default argument for `seed`. Throws if `inputSize < 1`.

### `addLayer(int neurons, const string& activationName)`
The new layer's input size = previous layer's output size (or `inputSize_` for the first
layer), so layers always fit together. Uses `makeActivation` (factory) to create the
activation object. Uses `emplace_back` to build the Layer directly inside the vector.

### `predict(const vector<double>& input)` → `vector<double>`
Checks the network has layers (`requireReady`) and the input has the right length,
then runs the private `forward`.

### `forward(const Matrix&)` (private)
Passes the data through each layer in order. Private because callers should use `predict`,
which validates input first — **encapsulation** of an unsafe internal step.

### `train(Dataset data, epochs, learningRate, reportEvery, ostream& log)` → final loss
Stochastic gradient descent:

```
for each epoch:
    shuffle samples                         (avoid learning the order)
    for each sample:
        output = forward(input)
        error  = output - target
        grad   = 2 * error / n              (derivative of mean squared error)
        for layer in reverse: grad = layer.backward(grad, lr)
    print loss every `reportEvery` epochs
```

Notes:
- `data` is taken **by value** (a copy) on purpose so shuffling doesn't reorder the
  caller's dataset.
- Uses **reverse iterators** (`rbegin()`/`rend()`) to walk layers backwards.
- Writes progress to any `ostream` — the App passes `std::cout`, the tests pass a
  silent `ostringstream`. This is association through an interface: the network doesn't
  care where the log goes.
- Validates shapes, empty dataset, epochs ≥ 1 and learning rate > 0.

### `evaluate(const Dataset&)` → `Evaluation{loss, accuracy}`
- **loss** = mean squared error over all outputs.
- **accuracy** = share of samples where *every* output, rounded to 0/1, equals the target.

### `save(filename)` / `static load(filename)` (file handling)
`save` writes with `max_digits10` precision so loading gives **exactly** the same weights.
Format:

```
MLP <inputSize> <layerCount>
<layer block>   (see Layer::save)
...
```

`load` is `static` because it *creates* a network — there's no network to call it on yet.
It checks the `MLP` header, that each layer's input size matches the previous layer's
output size, and that the file didn't end early. Any problem → `runtime_error`.

### `operator<<` (friend, operator overloading)
Prints the summary shown by menu option 6:

```
Input layer : 2 neuron(s)
Hidden 1    : 4 neuron(s), tanh, 12 parameters
Output layer: 1 neuron(s), sigmoid, 5 parameters
```

Declared `friend` so it can read the private `layers_` without adding public getters
just for printing. Parameter count per layer = `in × out` weights + `out` biases.

### Getters
`inputSize()`, `outputSize()` (last layer's size), `layerCount()`.
