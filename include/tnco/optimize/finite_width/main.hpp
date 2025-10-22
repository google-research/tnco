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

#include "cost_model/main.hpp"
#include "greedy/main.hpp"

namespace tnco::optimize::finite_width {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

void init(py::module &m) {
  // Initialize contraction costs
  {
    auto sm = m.def_submodule("cost_model");
    cost_model::init(sm);
  }

  // Initialize greedy optimization
  {
    auto sm = m.def_submodule("greedy");
    greedy::init(sm);
  }
}

}  // namespace tnco::optimize::finite_width
