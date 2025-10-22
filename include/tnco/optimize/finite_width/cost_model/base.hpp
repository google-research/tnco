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

namespace tnco::optimize::finite_width::cost_model::base {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename CostType, typename WidthType>
struct CostModel {
  using clone_ptr_type = std::unique_ptr<CostModel>;
  using cost_type = CostType;
  using width_type = WidthType;
  using bitset_type = tnco::bitset_type;
  using dims_type = tnco::ctree_type::dims_type;

  width_type max_width{};

  CostModel(width_type max_width) : max_width{std::move(max_width)} {
    // Max width must always be a positive number
    if (max_width < 0) {
      throw std::runtime_error(
          "'max_width' must always be a non-negative number");
    }
  }
  CostModel(const CostModel &) = default;
  CostModel(CostModel &&) = default;
  auto operator=(const CostModel &) -> CostModel & = delete;
  auto operator=(CostModel &&) -> CostModel & = delete;
  virtual ~CostModel() = default;

  [[nodiscard]] virtual auto width(const bitset_type &inds,
                                   const dims_type &dims) const -> width_type {
    /*
     * Return width.
     */
    return std::numeric_limits<width_type>::signaling_NaN();
  }

  [[nodiscard]] virtual auto delta_width(const bitset_type &inds,
                                         const dims_type &dims,
                                         const size_t &pos) const
      -> width_type {
    /*
     * Return the delta width as to add (if the index in position 'pos' is
     * absent) or remove (if index in position 'pos' is present).
     */
    return std::numeric_limits<width_type>::signaling_NaN();
  }

  [[nodiscard]] virtual auto contraction_cost(const bitset_type &inds_A,
                                              const bitset_type &inds_B,
                                              const bitset_type &inds_C,
                                              const dims_type &dims,
                                              const bitset_type &slices) const
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
  using width_type = typename self_type::width_type;
  py::class_<self_type>(m, name.c_str())
      .def(py::init<width_type>())
      .def(py::init<self_type>())
      .def(
          "__deepcopy__",
          [](const self_type &self, const py::dict &) -> auto {
            return self.clone();
          },
          "memo"_a)
      .def("__repr__",
           [](const self_type &self) -> auto {
             return "BaseCostModel(max_width=" +
                    tnco::to_string(self.max_width) +
                    ", width_type=" + type_to_str<width_type>() +
                    ", cost_type=" + type_to_str<cost_type>() + ")";
           })
      .def("__eq__",
           [](const self_type &self, const self_type &other) -> auto {
             return self.max_width == other.max_width;
           })
      .def_readonly("max_width", &self_type::max_width)
      .def("width", &self_type::width, "inds"_a, "dims"_a)
      .def("delta_width", &self_type::delta_width, "inds"_a, "dims"_a, "pos"_a)
      .def("contraction_cost", &self_type::contraction_cost, "inds_A"_a,
           "inds_B"_a, "inds_C"_a, "dims"_a, "slices"_a)
      .def_property_readonly("width_type",
                             [](const self_type &self) -> auto {
                               return type_to_str<width_type>();
                             })
      .def_property_readonly("cost_type", [](const self_type &self) -> auto {
        return type_to_str<cost_type>();
      });
}

}  // namespace tnco::optimize::finite_width::cost_model::base
