# Tensor Network Contraction Optimizer (TNCO)

[![python](https://img.shields.io/badge/python-3.8+-blue)]()
[![c++](https://img.shields.io/badge/c++-17-blue)]()
<br>
[![run_tests](https://github.com/google-research/tnco/actions/workflows/run_tests.yml/badge.svg)](https://github.com/google-research/tnco/actions/workflows/run_tests.yml)
[![cpp_linter](https://github.com/google-research/tnco/actions/workflows/cpp_linter.yml/badge.svg)](https://github.com/google-research/tnco/actions/workflows/cpp_linter.yml)
[![codeql](https://github.com/google-research/tnco/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/google-research/tnco/actions/workflows/github-code-scanning/codeql)
<br>
[![nature](https://img.shields.io/badge/Nature-https%3A%2F%2Fdoi.org%2F10.1038%2Fs41586--025--09526--6-darkblue)](https://doi.org/10.1038/s41586-025-09526-6)

`TNCO` is a heuristic tool that optimizes tensor network contraction paths. It
represents the contraction as a tree – with the initial tensors as leaves and
the final tensor as the root -- and explores possible paths by manipulating
this tree's structure. While the optimization is performed using simulated
annealing, the framework is extensible to other methods. `TNCO` supports
optimization with or without memory constraints, and can automatically
parallelize runs on multiple threads.

## Installation

### Prerequisites

Before installing `TNCO`, you must have the following system-level dependencies:

* C++17 compiler (`gcc >= 11`, `clang >= 13`)
* CMake (`cmake >= 3.5`)
* Python >= 3.8
* [boost::dynamic_bitset](https://github.com/boostorg/dynamic_bitset)
* GMP and MPFR (optional, for `float1024`)

### Install `TNCO` using `pip`

`TNCO` can be easily installed using `pip`:
```
pip install git+https://github.com/google-research/tnco
```
for the latest development version, or
```
pip install git+https://github.com/google-research/tnco@version
```
where `version` is one of the available
[versions](https://github.com/google-research/tnco/tags). `TNCO` can also be installed
from a [zip](https://github.com/google-research/tnco/archive/refs/heads/main.zip) file:
```
pip install tnco-main.zip
```
`TNCO` uses `joblib` to parallelize runs on multiple CPUs. This is an optional
dependency and is not installed by default. To install TNCO with joblib, use
the `[parallel]` extra:
```
pip install git+https://github.com/google-research/tnco[parallel]
```

### Install `TNCO` using `conda`

`TNCO` can also be installed using `conda` environments. Clone the `TNCO`
repository and execute the following from the project's root folder:
```
conda env create
```

### Install `TNCO` using `docker`

Finally, `TNCO` can also be installed in a `docker` container. Clone the `TNCO`
repository and execute the following from the project's root folder:
```
docker build . -t tnco
```

## How To Use

The library provides a user-friendly Python front-end for most common use
cases:
```
from tnco.app import Optimizer

tn = """
2 a b
2 b c
2 c d
"""

# Load optimizer
opt = Optimizer(method='sa')

# Perform the optimization
tn, res = opt.optimize(tn, betas=(0, 100), n_steps=100, n_runs=8)
```

Multiple formats are supported; see `tnco.app.load_tn` for further details. For
a more detailed example, see
[examples/Optimization.ipynb](examples/Optimization.ipynb).

The same front-end can be used from command line:
```
$ tnco optimize '[(2, "a", "b"), (2, "b", "c"), (2, "c", "d")]' \
                --betas='(0, 100)' \
                --n-steps=100 \
                --n-runs=8 \
                --verbose=10
```
For all the possible options run:
```
$ tnco --help
```

## Contact

TNCO was developed by [Salvatore Mandrà](https://github.com/s-mandra) in 2024.
For any questions or concerns not addressed here, please email
[smandra@google.com](mailto:smandra@google.com).

## Disclaimer

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

Copyright 2025 Google LLC.
