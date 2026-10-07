# 8. OOP & C++ Features Map

Where each requirement from the project brief is shown in the code. Use this to prepare for
the oral presentation — every row points to something you can open and explain.

## Basic C++ (brief §6)

| Feature | Where |
|---|---|
| `int`, `double`, `bool`, `string`, `char` | `Matrix::rows_` (int), `Matrix::data_` (double), `Evaluation`, `App::hasNetwork` (bool), `Dataset::name_` (string), `Matrix::requireSameShape(…, const char*)`, `'#'` / `','` in `Dataset::fromCsv` (char) |
| `cin` / `cout` | `App::readNumber`, `App::readLine`, every menu method |
| Arithmetic / relational / logical operators | `Activation.cpp` (sigmoid maths), `Matrix::operator()` bounds check (`r < 0 \|\| r >= rows_`) |
| `if` / `else` | `ReLU::activate`, validation checks everywhere |
| `switch` | `App::run`, `App::loadDataset` |
| `for` | `Matrix::operator*` (triple loop), `NeuralNetwork::train` |
| `while` | `Dataset::fromCsv` (reading lines), `App::readNumber` |
| `do-while` | `App::run` (main menu loop) |
| Functions with parameters / return values | everywhere, e.g. `NeuralNetwork::train` returns final loss |
| Function overloading | `Matrix::operator*(Matrix)` vs `operator*(double)`; `Dataset::addSample(vector, vector)` vs `addSample(Sample)`; two `Layer` constructors; three `Matrix` constructors |
| STL collections | `vector` (`Matrix::data_`, `Dataset::samples_`, `NeuralNetwork::layers_`) |

## OOP (brief §7)

| Concept | Where | Explanation |
|---|---|---|
| Classes & objects | 9 classes + 2 structs (see [architecture](01-architecture.md)) | `mlp::App app;` in `main.cpp` instantiates an object |
| Encapsulation | All data members are `private` (`Matrix::data_`, `Layer::weights_`, `NeuralNetwork::layers_`…) | If `Matrix::data_` were public, code could resize it and break `rows_ × cols_`; if `Layer::weights_` were public, anyone could change its shape and crash the next forward pass |
| Constructors | `Matrix(rows, cols, fill)`, `Layer(...)` ×2, `NeuralNetwork(inputSize, seed)`, `Dataset(name)`, `App()` | Constructors validate (throw on bad sizes) so an object can never exist in an invalid state |
| Abstraction | `ActivationFunction` (pure virtual `activate`, `derivative`, `name`) | Layer only knows "something that can activate and differentiate"; it cannot create a bare `ActivationFunction` |
| Inheritance | `Sigmoid`, `ReLU`, `Tanh` : `public ActivationFunction` | Genuine is-a: a sigmoid **is an** activation function |
| Polymorphism | `Layer::forward` / `Layer::backward` call `act.activate(z)` / `act.derivative(z)` through a base-class reference | The real object's version runs at run time |
| Composition | `NeuralNetwork` ◆ `Layer`; `Layer` ◆ `Matrix` + `ActivationFunction`; `Dataset` ◆ `Sample`; `App` ◆ `Dataset` + `NeuralNetwork` | Owner creates and destroys the parts. A network *has* layers — it is not a layer, so inheritance would be wrong |
| Association / dependency | `NeuralNetwork::train(Dataset, …)`, `evaluate(const Dataset&)`, `train(…, std::ostream& log)` | Network *uses* a dataset but does not own it |

## Advanced C++ (brief §9 — bonus)

| Feature | Bonus | Where |
|---|---|---|
| File persistence | +2 | `NeuralNetwork::save` / `load` (`ofstream` / `ifstream`), `Dataset::fromCsv` |
| Exception handling | +1 | Thrown: `invalid_argument`, `out_of_range`, `runtime_error`, `logic_error` across all classes. Caught: `App::run` (per action), `main`, `Dataset::fromCsv` (re-throws `stod` errors with line number) |
| Smart pointers | +2 | `unique_ptr<ActivationFunction>` in `Layer`; `unique_ptr<NeuralNetwork>` in `App`; `make_unique` in `makeActivation` |
| Templates | +2 | `App::readNumber<T>` (used with `int` and `double`); `throws<E>` helper in `tests/tests.cpp` |
| STL algorithms / advanced STL | +1 | `std::transform` (`Matrix`), `std::shuffle` (`Dataset`), `std::function` (`Matrix::apply`), reverse iterators (`NeuralNetwork::train`), iterator ranges (`Dataset::fromCsv`), `std::mt19937` + distributions |
| Operator overloading | +1 | `Matrix`: `()`, `+`, `-`, `*` ×2, `-=`, `<<`; `NeuralNetwork`: `<<` (friend) |
| Design pattern | +2 | **Factory**: `makeActivation(name)`; static factory methods `Dataset::xorGate()`, `NeuralNetwork::load()`. **Strategy**: the activation object is a swappable strategy inside each `Layer` |
| Lambdas | other | `Matrix` (`std::transform` lambdas), `Layer::forward/backward`, `App::buildNetwork` (`chooseActivation`) |
| Move semantics | other | `std::move` of `unique_ptr` and `Matrix` into `Layer`; `Layer` is move-only |
| `const` correctness | other | Getters and non-mutating methods are `const`; const and non-const `Matrix::operator()` |
| Namespaces | other | everything in `namespace mlp` |

## Error handling & validation (brief §11)

| Situation | Handling |
|---|---|
| Invalid menu choice / not a number | `readNumber` re-asks |
| Number out of range | `readNumber` re-asks with the allowed range |
| Empty text input | `readLine` re-asks |
| Action before network exists | `hasNetwork()` message |
| File cannot be opened | `runtime_error` → caught in `App::run` |
| Corrupt CSV / model file | `runtime_error` with reason → caught in `App::run` |
| Mismatched shapes (dataset vs network, input length) | `invalid_argument` → caught |
| Empty dataset | `invalid_argument` → caught |
| Unknown activation name | `invalid_argument` → caught |

## Likely oral questions — where to point

| Question (brief §17) | Show |
|---|---|
| 1. Class vs object | `class Layer` in `Layer.h` vs `layers_.emplace_back(...)` creating objects |
| 2. Encapsulation | `Layer` private members — explain what breaks if public |
| 3. Constructor | `Layer(int inputSize, int outputSize, unique_ptr<ActivationFunction>, mt19937&)` — each parameter |
| 4. Inheritance | `Sigmoid : public ActivationFunction` |
| 5. Polymorphism | `Layer::forward` — `act.activate(z)` through base reference |
| 6. Overloading vs overriding | Overloading: `Matrix::operator*` ×2, `Dataset::addSample` ×2. Overriding: `Sigmoid::activate override` |
| 7. Abstraction vs encapsulation | Abstraction = `ActivationFunction` hides *which* function; encapsulation = `private` hides *data* |
| 8. Composition | `NeuralNetwork` has a `vector<Layer>` — a network is not a layer |
| 9. STL choice | `vector`: ordered, index access, contiguous memory, works with `std::shuffle` |
| 10. Add a new derived class | See "How to add a new activation" in [03-activation-functions.md](03-activation-functions.md) |
