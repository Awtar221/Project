# 2. Matrix — `include/Matrix.h`, `src/Matrix.cpp`

## Purpose

All the maths of the network is matrix maths. `Matrix` is a small 2-D grid of `double`s
that supports exactly the operations the network needs — nothing more.

## Data (private — encapsulation)

| Member | Meaning |
|---|---|
| `rows_`, `cols_` | Shape |
| `data_` | `std::vector<double>` holding all values row by row (row-major) |

Element `(r, c)` lives at `data_[r * cols_ + c]`. Because `data_` is private, nobody can
make the shape and the data disagree (e.g. resize `data_` without changing `rows_`).

## Constructors

| Constructor | Used for |
|---|---|
| `Matrix()` | Empty placeholder (Layer's `lastInput_` / `lastZ_` before the first forward pass) |
| `Matrix(rows, cols, fill = 0.0)` | Zero matrix, e.g. biases. Throws `invalid_argument` if a size ≤ 0 |
| `explicit Matrix(const vector<double>&)` | Turn a list of inputs into a column vector (n × 1). `explicit` stops accidental conversions |
| `static random(rows, cols, range, rng)` | Random weights in `[-range, range]` |

## Operators (operator overloading)

| Operator | Meaning | Where it's used |
|---|---|---|
| `m(r, c)` | Read/write element, bounds-checked (`out_of_range`) | everywhere |
| `a + b` | Element-wise add | `W*x + b` in `Layer::forward` |
| `a - b` | Element-wise subtract | `output - target` in training |
| `a * b` (Matrix) | Matrix product | `W * x`, `Wᵀ * delta` |
| `a * 2.0` (double) | Scale every element | `gradient * learningRate` |
| `a -= b` | Subtract in place | weight update `weights_ -= ...` |
| `os << m` | Print the matrix | debugging / display |

`operator*` is **overloaded** twice — same name, different parameter type (`Matrix` vs
`double`). The compiler picks the right one from the argument.

## Other methods

- `hadamard(b)` — element-wise multiply (used in backprop: `gradient ⊙ activation'(z)`).
- `transpose()` — swaps rows and columns (`Wᵀ`, `xᵀ`).
- `apply(f)` — returns a new matrix with `f` applied to every element. Takes a
  `std::function`, so any **lambda** can be passed in (this is how activation functions are applied).
- `toVector()` — back to a plain `vector<double>`.
- `requireSameShape()` — private helper; throws if shapes differ.

## C++ features shown here

- STL algorithm `std::transform` with **lambdas** (`operator+`, `operator-`, `hadamard`, `apply`)
- **Exceptions**: `invalid_argument` for bad shapes, `out_of_range` for bad indices
- `const` correctness: every method that doesn't change the matrix is `const`; there are
  two versions of `operator()` (const and non-const)
