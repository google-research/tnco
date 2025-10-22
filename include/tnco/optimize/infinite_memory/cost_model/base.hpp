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

#include <limits>
#include <memory>
#include <tnco/globals.hpp>
#include <variant>
#include <vector>

namespace tnco::optimize::infinite_memory::cost_model::base {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename CostType>
struct CostModel {
  using clone_ptr_type = std::unique_ptr<CostModel>;
  using cost_type = CostType;
  using bitset_type = tnco::bitset_type;
  using dims_type = tnco::ctree_type::dims_type;

  CostModel() = default;
  CostModel(const CostModel &) = default;
  CostModel(CostModel &&) = default;
  auto operator=(const CostModel &) -> CostModel & = delete;
  auto operator=(CostModel &&) -> CostModel & = delete;
  virtual ~CostModel() = default;

  [[nodiscard]] virtual auto contraction_cost(const bitset_type &inds_A,
                                              const bitset_type &inds_B,
                                              const bitset_type &inds_C,
                                              const dims_type &dims) const
      -> cost_type {
    /*
     * Compute the contraction cost of:
     *
     *    A
     *   / \
     *  B   C
     *
     */
    return std::numeric_limits<cost_type>::signaling_NaN();
  }

  [[nodiscard]] virtual auto clone() const noexcept -> clone_ptr_type {
    return std::make_unique<CostModel>(*this);
  }
};

template <typename... T>
void init(py::module &m, const std::string &name) {
  using self_type = CostModel<T...>;
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
             return "BaseCostModel(cost_type=" + type_to_str<cost_type>() + ")";
           })
      .def("__eq__",
           [](const self_type &self, const self_type &other) -> auto {
             return true;
           })
      .def("contraction_cost", &self_type::contraction_cost, "inds_A"_a,
           "inds_B"_a, "inds_C"_a, "dims"_a)
      .def_property_readonly("cost_type", [](const self_type &self) -> auto {
        return type_to_str<cost_type>();
      });
}

}  // namespace tnco::optimize::infinite_memory::cost_model::base
