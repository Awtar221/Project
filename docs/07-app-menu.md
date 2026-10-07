# 7. App (Console Menu) — `include/App.h`, `src/App.cpp`, `main.cpp`

## Purpose

The only class that talks to the user. It reads input, calls `Dataset` / `NeuralNetwork`,
and prints results. No maths happens here.

## `main.cpp`

```cpp
mlp::App app;
app.run();
```

Wrapped in `try/catch` as a last safety net — if something unexpected escapes, the program
prints `Fatal error: ...` instead of crashing silently.

## Data (private)

| Member | Meaning |
|---|---|
| `dataset_` | Current `Dataset` (XOR loaded by default in the constructor) |
| `network_` | `std::unique_ptr<NeuralNetwork>` — **null until** the user builds or loads one |

A `unique_ptr` is used for the network because "no network yet" is a real state, and
rebuilding simply assigns a new one (the old one is freed automatically).

## The main loop — `run()`

```cpp
do {
    showMenu();
    choice = readNumber<int>("Choose an option: ", 0, 8);
    try {
        switch (choice) { case 1: loadDataset(); break; ... }
    } catch (const std::exception& e) {
        std::cout << "  Error: " << e.what() << '\n';
    }
} while (choice != 0);
```

- `do-while` because the menu must show at least once.
- `switch` dispatches to one private method per menu option.
- The `try/catch` around the switch is the **central error handler**: any exception thrown
  deep inside (`Matrix`, `Dataset::fromCsv`, `NeuralNetwork::load`, …) arrives here as a
  message, and the user is back at the menu.

## Menu actions

| Option | Method | What it does |
|---|---|---|
| 1 | `loadDataset()` | Built-in gate or CSV; warns if the current network no longer fits |
| 2 | `buildNetwork()` | Asks hidden layers / neurons / activations; output layer size comes from the dataset |
| 3 | `trainNetwork()` | Asks epochs + learning rate, logs loss 10 times |
| 4 | `evaluateNetwork()` | Prints every input → output (target), then MSE + accuracy |
| 5 | `predictInput()` | Asks one number per input, prints output |
| 6 | `showSummary()` | `std::cout << *network_` (uses overloaded `operator<<`) |
| 7 | `saveModel()` | Asks path, calls `NeuralNetwork::save` |
| 8 | `loadModel()` | Asks path, calls `NeuralNetwork::load` |

`hasDataset()` / `hasNetwork()` print a friendly message and return `false` when a step is
attempted too early (e.g. training before building).

In `buildNetwork()`, a **lambda** `chooseActivation` captures the list of names and is reused
for every layer. The network is built into a local `unique_ptr` first and only moved into
`network_` once complete — so if something fails half-way, the old network is kept.

## Input validation — `readNumber<T>` (template)

```cpp
template <typename T>
static T readNumber(const std::string& prompt, T min, T max);
```

- One function works for `int` (menu choices, epochs) and `double` (learning rate, inputs).
- Loops until the value parses **and** is within `[min, max]`.
- On bad input: `cin.clear()` resets the error flag, `cin.ignore(...)` throws away the bad
  line, then it asks again — so typing `abc` never breaks the program.
- If input ends (Ctrl+Z / Ctrl+D) it throws instead of looping forever.

It's in the header because templates must be visible where they're used.

`readLine(prompt)` does the same for text (file paths) — refuses empty input.
