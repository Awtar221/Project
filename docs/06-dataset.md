# 6. Dataset — `include/Dataset.h`, `src/Dataset.cpp`

## Purpose

Holds the examples the network learns from.

## `Sample` (struct)

```cpp
struct Sample {
    std::vector<double> input;
    std::vector<double> target;
};
```

A plain data holder — no rules of its own, so a `struct` with public fields is appropriate.
The *rules* (all samples must have the same size) are enforced by `Dataset`.

## `Dataset` data (private)

| Member | Meaning |
|---|---|
| `name_` | Shown in the menu ("XOR", or the CSV path) |
| `samples_` | `std::vector<Sample>` — vector because we need order, indexing and shuffling |

## Built-in datasets (static factory methods)

`xorGate()`, `andGate()`, `orGate()` — each returns a 4-sample dataset with 2 inputs and 1
output. `static` because they create a new Dataset rather than act on an existing one.

| x1 | x2 | XOR | AND | OR |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 | 1 |
| 1 | 0 | 1 | 0 | 1 |
| 1 | 1 | 0 | 1 | 1 |

AND and OR are linearly separable (easy); XOR is not, which is why a hidden layer is needed.

## `fromCsv(filename, outputCount)` (file handling + exceptions)

1. Opens the file with `std::ifstream` → `runtime_error` if it can't.
2. Reads line by line with `std::getline`; skips blank lines and `#` comments.
3. Splits each line on `,` using a `std::stringstream`.
4. Converts each cell with `std::stod`; a bad cell → `runtime_error("Line N: 'x' is not a number")`.
5. The last `outputCount` columns become the target, the rest the input
   (split using **iterators**: `values.end() - outputCount`).
6. Throws if a row has too few columns or the file has no data.

## `addSample` (function overloading)

```cpp
void addSample(const std::vector<double>& input, const std::vector<double>& target);
void addSample(const Sample& sample);
```

Same name, different parameters. The first is convenient for the built-in gates
(`d.addSample({0, 1}, {1})`), the second accepts a ready-made `Sample`. The first simply
forwards to the second, so **validation lives in one place**:

- input and target must not be empty
- every sample must have the same input/output size as the first one

## `shuffle(rng)`

Uses the STL algorithm `std::shuffle`. Called once per epoch during training.

## Getters

`name()`, `samples()` (const reference — no copy, can't be modified from outside), `size()`,
`empty()`, `inputSize()`, `outputSize()` (0 when empty).
