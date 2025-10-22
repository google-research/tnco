# Tensor Network Contraction Optimizer (TNCO)

`TNCO` is a heuristic tool that optimizes tensor network contraction paths. It
represents the contraction as a tree -- with the initial tensors as leaves and
the final tensor as the root -- and explores possible paths by manipulating
this tree's structure. While the optimization is performed using simulated
annealing, the framework is extensible to other methods. `TNCO` supports
optimization with or without memory constraints, and can automatically
parallelize runs on multiple threads.

## How To Install

The easiest way to install `TNCO` is using `pip`:
```
pip install [tnco_folder]
```
`TNCO` uses `joblib` to parallelize runs on multiple CPUs (which is not installed
by default). It can be automatically installed by installing the extra
dependencies `[parallel]`.

### Requires

* C++17 compiler (`gcc >= 11`, `clang >= 13`)
* CMake (`cmake >= 3.5`)
* Python >= 3.8
* [pybind11](https://github.com/pybind/pybind11)
* [boost::dynamic_bitset](https://github.com/boostorg/dynamic_bitset)
* GMP and MPFR (optional, for `float1024`)

Excluding the C++17 compiler and CMake, all the dependencies (excluding the
compiler, CMake, Python, and the optional requires) can be automatically
downloaded and installed by setting `DOWNLOAD_DEPS=1` as an env variable, for
instance:
```
DOWNLOAD_DEPS=1 pip install [tnco_folder]
```

## How To Use

The library provides a user-friendly Python front-end for most of the common
uses:
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
Multiple formats are supported, see `tnco.app.load_tn` for further details.
For a more detailed example, see [examples/Optimization.ipynb](https://github.com/google-research/tnco/blob/main/examples/Optimization.ipynb).

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
