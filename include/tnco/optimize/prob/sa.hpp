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

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <tnco/assert.hpp>
#include <tnco/globals.hpp>

#include "base.hpp"

namespace tnco::optimize::prob::sa {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename... T>
struct Probability final : prob::base::Probability<T...> {
  using base_type = prob::base::Probability<T...>;
  using clone_ptr_type = typename base_type::clone_ptr_type;
  using cost_type = typename base_type::cost_type;
  using beta_type = tnco::beta_type;

  Probability(beta_type beta = 0) : beta{beta} {}

  beta_type beta{};

  [[nodiscard]] auto operator()(const cost_type &delta_cost,
                                const cost_type &old_cost) const
#ifdef NDEBUG
      noexcept
#endif
      -> cost_type override {
    ASSERT(delta_cost <= 0 || (delta_cost / old_cost) >= -1, "Domain error.");
    return delta_cost <= 0 ? cost_type{1}
                           : pow(1 + (delta_cost / old_cost), -beta);
  }

  [[nodiscard]] auto clone() const noexcept -> clone_ptr_type override {
    return std::make_unique<Probability>(*this);
  }
};

template <typename... T>
void init(py::module &m, const std::string &name) {
  using self_type = Probability<T...>;
  using base_type = typename self_type::base_type;
  using cost_type = typename self_type::cost_type;
  using beta_type = typename self_type::beta_type;
  py::class_<self_type, base_type>(m, name.c_str())
      .def(py::init<>())
      .def(py::init<beta_type>())
      .def(py::init<self_type>())
      .def_readwrite("beta", &self_type::beta)
      .def("__repr__",
           [](const self_type &self) -> auto {
             return "SimulatedAnnealing(beta=" + tnco::to_string(self.beta) +
                    ", cost_type=" + type_to_str<cost_type>() + ")";
           })
      .def("__eq__",
           [](const self_type &self, const self_type &other) -> auto {
             return self.beta == other.beta;
           })
      .def(py::pickle(
          [](const self_type &self) -> auto { return std::tuple{self.beta}; },
          [](const std::tuple<decltype(self_type::beta)> &state) -> auto {
            return self_type{std::get<0>(state)};
          }));
}

}  // namespace tnco::optimize::prob::sa
