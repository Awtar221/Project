# Simple MLP — Multi-Layer Perceptron in C++

A console application that lets you build, train, test, save and load a small
neural network (a Multi-Layer Perceptron) written from scratch in C++17 —
no external libraries.

Built for **BAI22123 Object-Oriented Programming** (Raffles University, Sept 2026).

---

## What it does

- Load a dataset — built-in logic gates (**XOR**, **AND**, **OR**) or your own **CSV file**
- Build a network — choose 1–5 hidden layers, neurons per layer, and an activation
  function (**sigmoid**, **ReLU**, **tanh**) for each layer
- Train it with backpropagation + stochastic gradient descent (mean squared error loss)
- Evaluate it — see each prediction next to its target, plus loss and accuracy
- Predict any custom input
- Save the trained model to a text file and load it back later

---

## Project structure

```
Project/
├── main.cpp                 Program entry point — creates App and runs it
├── include/                 Header files (class declarations)
│   ├── App.h                Console menu
│   ├── NeuralNetwork.h      The MLP (stack of layers)
│   ├── Layer.h              One dense layer
│   ├── Activation.h         Abstract ActivationFunction + Sigmoid / ReLU / Tanh
│   ├── Dataset.h            Training data (Sample + Dataset)
│   └── Matrix.h             Small matrix maths class
├── src/                     Source files (class implementations, same names as headers)
├── tests/tests.cpp          17 automated test cases
├── data/xor.csv             Example CSV dataset
├── diagrams/                UML diagrams (open with draw.io)
│   ├── use_case_diagram.drawio
│   └── class_diagram.drawio
└── docs/                    Explanation of every part of the code
```

---

## Requirements

- A C++17 compiler — any one of:
  - **g++** 7+ (MinGW-w64 / MSYS2 on Windows, or Linux/macOS)
  - **MSVC** (Visual Studio 2019/2022 or Build Tools)
  - **clang++** 5+
- No external libraries.

---

## How to compile

Run these from the `Project/` folder.

### Option A — g++ (MinGW / Linux / macOS)

```bash
mkdir build
g++ -std=c++17 -Wall -Iinclude main.cpp src/*.cpp -o build/mlp
```

Tests:

```bash
g++ -std=c++17 -Wall -Iinclude tests/tests.cpp src/Matrix.cpp src/Activation.cpp src/Layer.cpp src/Dataset.cpp src/NeuralNetwork.cpp -o build/tests
```

### Option B — MSVC (Windows)

Open **"x64 Native Tools Command Prompt for VS"** (Start menu), `cd` into `Project/`, then:

```bat
mkdir build
cl /std:c++17 /EHsc /W4 /Iinclude main.cpp src\*.cpp /Fo:build\ /Fe:build\mlp.exe
cl /std:c++17 /EHsc /W4 /Iinclude tests\tests.cpp src\Matrix.cpp src\Activation.cpp src\Layer.cpp src\Dataset.cpp src\NeuralNetwork.cpp /Fo:build\ /Fe:build\tests.exe
```

### Option C — Visual Studio IDE

Create an empty C++ console project, add `main.cpp` and everything in `src/`,
add `include/` under *Project → Properties → C/C++ → Additional Include Directories*,
set *C++ Language Standard* to C++17, then build and run.

---

## How to run

Run from the `Project/` folder (so the `data/` paths work):

```bash
./build/mlp          # Linux / macOS / Git Bash
build\mlp.exe        # Windows cmd / PowerShell
```

Run the tests:

```bash
./build/tests        # or build\tests.exe
```

Expected last line: `All tests passed.`

---

## Using the program

The XOR dataset is loaded automatically at start-up.

```
--------------- MAIN MENU ---------------
 Dataset: XOR (4 samples)   Network: not built
 1. Load dataset
 2. Build network
 3. Train network
 4. Evaluate network
 5. Predict a custom input
 6. Show network summary
 7. Save model to file
 8. Load model from file
 0. Exit
```

| Option | What it does | Input asked for |
|---|---|---|
| 1 | Pick XOR / AND / OR, or load a CSV | CSV path + number of output columns |
| 2 | Build a new network for the current dataset | hidden layers (1–5), neurons each (1–64), activation per layer |
| 3 | Train the network | epochs (1–100000), learning rate (0.0001–10) |
| 4 | Show every prediction vs target, MSE loss and accuracy | — |
| 5 | Run the network on numbers you type | one number per input neuron |
| 6 | Print layer sizes, activations, parameter counts | — |
| 7 | Save weights to a text file | file path, e.g. `data/model.txt` |
| 8 | Load weights from a text file | file path |
| 0 | Quit | — |

### Quick demo (learn XOR)

1. `2` → hidden layers `1` → neurons `4` → hidden activation `3` (tanh) → output activation `1` (sigmoid)
2. `3` → epochs `3000` → learning rate `0.5`
3. `4` → you should see outputs close to 0, 1, 1, 0 and **Accuracy: 100%**
4. `7` → `data/model.txt` to save, `8` → `data/model.txt` to load it back

Recommended settings: **tanh hidden + sigmoid output, lr 0.5, 3000+ epochs**.
ReLU hidden layers can occasionally get stuck on XOR ("dead neurons") — rebuild and retrain.

### CSV format

One sample per line, comma-separated numbers. The **last N columns are targets**
(you enter N when loading). Blank lines and lines starting with `#` are ignored.

```
# x1,x2,target
0,0,0
0,1,1
1,0,1
1,1,0
```

### Model file format

```
MLP <inputSize> <layerCount>
<in> <out> <activation>      ← one block per layer
<weights row by row>
<biases>
```

---

## Error messages

The program never crashes on bad input — it prints a message and returns to the menu.

| Message | Cause |
|---|---|
| `Invalid input. Enter a value between X and Y.` | Not a number, or out of range |
| `No network yet. Use option 2 (build) or 8 (load) first.` | Trained/evaluated/saved before building |
| `Error: Cannot open file: ...` | CSV or model file path wrong |
| `Error: Line N: 'x' is not a number` | Bad cell in CSV |
| `Error: Sample size does not match the rest of the dataset` | CSV rows have different column counts |
| `Error: Not a valid model file: ...` | Loaded a file that isn't a saved model |
| `Error: Dataset shape does not match network shape` | Dataset changed after building — rebuild (option 2) |
| `Error: Expected N inputs, got M` | Wrong number of inputs for prediction |

---

## Documentation

See [`docs/`](docs/) for a walkthrough of every class and where each OOP / C++ requirement lives in the code:

1. [Architecture overview](docs/01-architecture.md)
2. [Matrix](docs/02-matrix.md)
3. [Activation functions](docs/03-activation-functions.md)
4. [Layer](docs/04-layer.md)
5. [NeuralNetwork](docs/05-neural-network.md)
6. [Dataset](docs/06-dataset.md)
7. [App (menu)](docs/07-app-menu.md)
8. [OOP & C++ features map](docs/08-oop-cpp-features.md)
9. [Test cases](docs/09-test-cases.md)

UML diagrams: open the files in `diagrams/` with [draw.io](https://app.diagrams.net)
(*File → Open from → Device*) or the draw.io desktop app / VS Code extension.
