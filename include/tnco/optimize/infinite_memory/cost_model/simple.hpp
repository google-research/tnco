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
#include <memory>
#include <tnco/assert.hpp>
#include <tnco/globals.hpp>
#include <variant>
#include <vector>

#include "base.hpp"

namespace tnco::optimize::infinite_memory::cost_model::simple {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename CostType, typename Bitset, typename DimsType>
[[nodiscard]] auto get_cost(const Bitset &inds, const DimsType &dims)
#ifdef NDEBUG
    noexcept
#endif
    -> CostType {
  if constexpr (std::is_arithmetic_v<DimsType>) {
    ASSERT(dims != 0, "Each dimension must be a positive number.");
    return std::pow(dims, inds.count());
  } else {
    ASSERT(std::size(inds) <= std::size(dims), "'Too few dimensions.'");
    ASSERT(std::none_of(std::begin(dims), std::end(dims),
                        [](auto &&d) -> auto { return d == 0; }),
           "Each dimension must be a positive number.");
    CostType cost_{1};
    inds.visit([&cost_, &dims](auto &&pos) -> auto { cost_ *= dims[pos]; });
    return cost_;
  }
}

template <typename... T>
struct CostModel : cost_model::base::CostModel<T...> {
  using base_type = cost_model::base::CostModel<T...>;
  using clone_ptr_type = typename base_type::clone_ptr_type;
  using cost_type = typename base_type::cost_type;
  using bitset_type = typename base_type::bitset_type;
  using dims_type = typename base_type::dims_type;

  [[nodiscard]] auto contraction_cost(const bitset_type &inds_A,
                                      const bitset_type &inds_B,
                                      const bitset_type &inds_C,
                                      const dims_type &dims) const
#ifdef NDEBUG
      noexcept
#endif
      -> cost_type override {
    ASSERT(std::size(inds_A) == std::size(inds_B) &&
               std::size(inds_A) == std::size(inds_C),
           "Indices must have the same size.");
    ASSERT(inds_C.is_subset_of(inds_A | inds_B),
           "'inds_C' must be a subset of 'inds_A | inds_B'.");
    return std::visit(
        [inds = inds_A | inds_B](auto &&dims) -> auto {
          return get_cost<cost_type>(inds, dims);
        },
        dims);
  }

  [[nodiscard]] auto clone() const noexcept -> clone_ptr_type override {
    return std::make_unique<CostModel>(*this);
  }
};

template <typename... T>
void init(py::module &m, const std::string &name) {
  using self_type = CostModel<T...>;
  using base_type = typename self_type::base_type;
  using cost_type = typename self_type::cost_type;
  py::class_<self_type, base_type>(m, name.c_str())
      .def(py::init<>())
      .def(py::init<self_type>())
      .def("__repr__",
           [](const self_type &self) -> auto {
             return "SimpleCostModel(cost_type=" + type_to_str<cost_type>() +
                    ")";
           })
      .def("__eq__",
           [](const self_type &self, const self_type &other) -> auto {
             return true;
           })
      .def(
          py::pickle([](const self_type &self) -> auto { return std::tuple{}; },
                     [](const std::tuple<> &) -> auto { return self_type{}; }));
}

}  // namespace tnco::optimize::infinite_memory::cost_model::simple
