// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "finite_width/main.hpp"
#include "infinite_memory/main.hpp"
#include "optimizer.hpp"
#include "prob/main.hpp"

namespace tnco::optimize {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

void init(py::module &m) {
  // Initialize probabilities
  {
    auto sm = m.def_submodule("prob");
    prob::init(sm);
  }

  // Initialize base optimizer
  optimizer::init(m, "BaseOptimizer");

  // Initialize infinite memory
  {
    auto sm = m.def_submodule("infinite_memory");
    infinite_memory::init(sm);
  }

  // Initialize finite_with
  {
    auto sm = m.def_submodule("finite_width");
    finite_width::init(sm);
  }
}

}  // namespace tnco::optimize
