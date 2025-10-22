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

#include <memory>
#include <tnco/globals.hpp>

namespace tnco::optimize::prob::base {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename CostType>
struct Probability {
  using clone_ptr_type = std::unique_ptr<Probability>;
  using cost_type = CostType;

  Probability() = default;
  Probability(const Probability &) = default;
  Probability(Probability &&) = default;
  auto operator=(const Probability &) -> Probability & = delete;
  auto operator=(Probability &&) -> Probability & = delete;
  virtual ~Probability() = default;

  [[nodiscard]] virtual auto operator()(const cost_type &delta_cost,
                                        const cost_type &old_cost) const
      -> cost_type {
    return 1;
  }

  [[nodiscard]] virtual auto clone() const noexcept -> clone_ptr_type {
    return std::make_unique<Probability>(*this);
  }
};

template <typename... T>
void init(py::module &m, const std::string &name) {
  using self_type = Probability<T...>;
  using cost_type = typename self_type::cost_type;
  py::class_<self_type>(m, name.c_str())
      .def(py::init<>())
      .def(py::init<self_type>())
      .def(
          "__deepcopy__",
          [](const self_type &self, const py::dict &) -> auto {
            return self.clone();
          },
          "memo"_a)
      .def("__repr__",
           [](const self_type &self) -> auto {
             return "BaseProbability(cost_type=" + type_to_str<cost_type>() +
                    ")";
           })
      .def("__eq__",
           [](const self_type &self, const self_type &other) -> auto {
             return true;
           })
      .def("__call__", &self_type::operator(), "delta_cost"_a, "old_cost"_a)
      .def(
          py::pickle([](const self_type &self) -> auto { return std::tuple{}; },
                     [](const std::tuple<> &) -> auto { return self_type{}; }))
      .def_property_readonly("cost_type", [](const self_type &self) -> auto {
        return type_to_str<cost_type>();
      });
}

}  // namespace tnco::optimize::prob::base
