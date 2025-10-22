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

#include <tnco/globals.hpp>

#include "base.hpp"
#include "greedy.hpp"
#include "sa.hpp"

namespace tnco::optimize::prob {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename CostType>
auto core(py::module &m) -> void {
  // Get suffix
  const std::string sfx = "_" + type_to_str<CostType>();

  // Initialize
  base::init<CostType>(m, "BaseProbability" + sfx);
  greedy::init<CostType>(m, "Greedy" + sfx);
  sa::init<CostType>(m, "SimulatedAnnealing" + sfx);
}

auto init(py::module &m) -> void { EXPAND_COST_TYPE(core, m); }

}  // namespace tnco::optimize::prob
