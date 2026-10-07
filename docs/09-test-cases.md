# 9. Test Cases

## Automated tests — `tests/tests.cpp`

Build and run (see README). Each test prints `PASS` or `FAIL`. Results below are from the
last run (MSVC 19.44, C++17).

| ID | Description | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| T01 | Matrix product | `[[1,2],[3,4]] × [1,1]ᵀ` | `[3, 7]ᵀ` | `[3, 7]ᵀ` | PASS |
| T02 | Matrix product, wrong shapes | `(2×3) × (2×3)` | `invalid_argument` thrown | thrown | PASS |
| T03 | Transpose | 2×3 matrix | 3×2 matrix | 3×2 | PASS |
| T04 | Index out of range | `Matrix(2,2)(5,0)` | `out_of_range` thrown | thrown | PASS |
| T05 | Scalar multiply & Hadamard | `[3,3] × 2`, `[3,3] ⊙ [3,3]` | `6`, `9` | `6`, `9` | PASS |
| T06 | Sigmoid values | `z = 0` | `0.5`, derivative `0.25` | `0.5`, `0.25` | PASS |
| T07 | ReLU values | `z = -3, 2, -1` | `0, 2`, derivative `0` | `0, 2, 0` | PASS |
| T08 | Polymorphism via base pointer | `makeActivation("tanh")` | name `tanh`, `f(0)=0`, `f'(0)=1` | as expected | PASS |
| T09 | Unknown activation | `makeActivation("banana")` | `invalid_argument` thrown | thrown | PASS |
| T10 | Built-in XOR dataset | `Dataset::xorGate()` | 4 samples, 2 in, 1 out | 4, 2, 1 | PASS |
| T11 | Mismatched sample | add `{1,2,3}` to XOR | `invalid_argument` thrown | thrown | PASS |
| T12 | Missing CSV file | `fromCsv("no_such_file.csv")` | `runtime_error` thrown | thrown | PASS |
| T13 | Predict with no layers | `NeuralNetwork(2).predict({0,0})` | `logic_error` thrown | thrown | PASS |
| T14 | Predict wrong input count | 3 inputs to a 2-input net | `invalid_argument` thrown | thrown | PASS |
| T15 | Network learns XOR | 2-4(tanh)-1(sigmoid), 5000 epochs, lr 0.5 | accuracy 100%, loss < 0.05 | 100% | PASS |
| T16 | Save/load round-trip | save then load a 2-3-1 net | identical prediction | identical | PASS |
| T17 | Load non-model file | `load("data/xor.csv")` | `runtime_error` thrown | thrown | PASS |

## Manual menu tests

Run `build/mlp` and type the inputs shown.

| ID | Description | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| M01 | Invalid menu number | `9` | `Invalid input. Enter a value between 0 and 8.` | as expected | PASS |
| M02 | Non-numeric menu input | `abc` | Same message, menu asks again | as expected | PASS |
| M03 | Train before building | `3` | `No network yet. Use option 2 (build) or 8 (load) first.` | as expected | PASS |
| M04 | Build XOR network | `2`, `1`, `4`, `3`, `1` | Summary: hidden 4 tanh, output 1 sigmoid | as expected | PASS |
| M05 | Train XOR | `3`, `3000`, `0.5` | Loss printed 10×, decreasing | 0.0015 → 0.0001 | PASS |
| M06 | Evaluate XOR | `4` | Outputs ≈ 0,1,1,0; Accuracy 100% | 0.004, 0.990, 0.989, 0.012; 100% | PASS |
| M07 | Custom predict | `5`, `1`, `0` | Output ≈ 1 | 0.9892 | PASS |
| M08 | Save model | `7`, `data/model.txt` | `Model saved to data/model.txt` | as expected | PASS |
| M09 | Load CSV — missing file | `1`, `4`, `nope.csv`, `1` | `Error: Cannot open file: nope.csv` | as expected | PASS |
| M10 | Load CSV — valid | `1`, `4`, `data/xor.csv`, `1` | Loaded 4 samples, 2 inputs, 1 output | as expected | PASS |
| M11 | Load saved model | `8`, `data/model.txt` | Same summary and same predictions as before save | as expected | PASS |
| M12 | Load invalid model | `8`, `data/xor.csv` | `Error: Not a valid model file: data/xor.csv` | as expected | PASS |
| M13 | Exit | `0` | `Goodbye!`, program ends | as expected | PASS |
