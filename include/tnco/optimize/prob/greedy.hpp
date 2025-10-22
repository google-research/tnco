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

#include <tnco/globals.hpp>

#include "base.hpp"

namespace tnco::optimize::prob::greedy {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename... T>
struct Probability final : prob::base::Probability<T...> {
  using base_type = prob::base::Probability<T...>;
  using clone_ptr_type = typename base_type::clone_ptr_type;
  using cost_type = typename base_type::cost_type;

  [[nodiscard]] auto operator()(const cost_type &delta_cost,
                                const cost_type &old_cost) const noexcept
      -> cost_type override {
    return delta_cost <= 0 ? 1 : 0;
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
  py::class_<self_type, base_type>(m, name.c_str())
      .def(py::init<>())
      .def(py::init<self_type>())
      .def("__repr__",
           [](const self_type &self) -> auto {
             return "Greedy(cost_type=" + type_to_str<cost_type>() + ")";
           })
      .def("__eq__",
           [](const self_type &self, const self_type &other) -> auto {
             return true;
           })
      .def(
          py::pickle([](const self_type &self) -> auto { return std::tuple{}; },
                     [](const std::tuple<> &) -> auto { return self_type{}; }));
}

}  // namespace tnco::optimize::prob::greedy
