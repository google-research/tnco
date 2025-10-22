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
#include <tnco/optimize/infinite_memory/cost_model/simple.hpp>
#include <variant>
#include <vector>

#include "base.hpp"

namespace tnco::optimize::finite_width::cost_model::simple {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename WidthType, typename Bitset, typename DimsType>
[[nodiscard]] auto get_width(const Bitset &inds, const DimsType &dims)
#ifdef NDEBUG
    noexcept
#endif
    -> WidthType {
  if constexpr (std::is_arithmetic_v<DimsType>) {
    ASSERT(dims != 0, "Each dimension must be a positive number.");
    return log2(dims) * inds.count();
  } else {
    ASSERT(std::size(inds) <= std::size(dims), "'Too few dimensions.'");
    ASSERT(std::none_of(std::begin(dims), std::end(dims),
                        [](auto &&d) -> auto { return d == 0; }),
           "Each dimension must be a positive number.");
    WidthType width_{0};
    inds.visit(
        [&width_, &dims](auto &&pos) -> auto { width_ += log2(dims[pos]); });
    return width_;
  }
}

template <typename WidthType, typename Bitset, typename DimsType>
[[nodiscard]] auto get_delta_width(const Bitset &inds, const DimsType &dims,
                                   const size_t &pos)
#ifdef NDEBUG
    noexcept
#endif
    -> WidthType {
  if constexpr (std::is_arithmetic_v<DimsType>) {
    ASSERT(dims != 0, "Each dimension must be a positive number.");
    return (1 - 2 * inds.test(pos)) * log2(dims);
  } else {
    ASSERT(std::size(inds) <= std::size(dims), "'Too few dimensions.'");
    ASSERT(std::none_of(std::begin(dims), std::end(dims),
                        [](auto &&d) -> auto { return d == 0; }),
           "Each dimension must be a positive number.");
    return (1 - 2 * inds.test(pos)) * log2(dims[pos]);
  }
}

template <typename... T>
struct CostModel : cost_model::base::CostModel<T...> {
  using base_type = cost_model::base::CostModel<T...>;
  using clone_ptr_type = typename base_type::clone_ptr_type;
  using cost_type = typename base_type::cost_type;
  using width_type = typename base_type::width_type;
  using bitset_type = typename base_type::bitset_type;
  using dims_type = typename base_type::dims_type;

  CostModel(width_type max_width) : base_type{std::move(max_width)} {
    if (max_width < 0) {
      throw std::runtime_error("'max_width' must be a non-negative number.");
    }
  }

  [[nodiscard]] auto width(const bitset_type &inds, const dims_type &dims) const
#ifdef NDEBUG
      noexcept
#endif
      -> width_type override {
    /*
     * Return width.
     */
    return std::visit(
        [&inds](auto &&dims) -> auto {
          return get_width<width_type>(inds, dims);
        },
        dims);
  }

  [[nodiscard]] auto delta_width(const bitset_type &inds, const dims_type &dims,
                                 const size_t &pos) const
#ifdef NDEBUG
      noexcept
#endif
      -> width_type override {
    /*
     * Return the delta width as to add / remove index in position 'pos'.
     */
    return std::visit(
        [&inds, &pos](auto &&dims) -> auto {
          return get_delta_width<width_type>(inds, dims, pos);
        },
        dims);
  }

  [[nodiscard]] auto contraction_cost(const bitset_type &inds_A,
                                      const bitset_type &inds_B,
                                      const bitset_type &inds_C,
                                      const dims_type &dims,
                                      const bitset_type &slices) const
#ifdef NDEBUG
      noexcept
#endif
      -> cost_type override {
    ASSERT(std::size(inds_A) == std::size(inds_B) &&
               std::size(inds_A) == std::size(inds_C) &&
               std::size(inds_A) == std::size(slices),
           "Indices  and slices must have the same size.");
    ASSERT(inds_C.is_subset_of(inds_A | inds_B),
           "'inds_C' must be a subset of 'inds_A | inds_B'.");
    return std::visit(
        [inds = inds_A | inds_B | slices](auto &&dims) -> auto {
          return infinite_memory::cost_model::simple::get_cost<cost_type>(inds,
                                                                          dims);
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
  using width_type = typename self_type::width_type;
  py::class_<self_type, base_type>(m, name.c_str())
      .def(py::init<width_type>())
      .def(py::init<self_type>())
      .def("__repr__",
           [](const self_type &self) -> auto {
             return "SimpleCostModel(max_width=" +
                    tnco::to_string(self.max_width) +
                    ", width_type=" + type_to_str<width_type>() +
                    ", cost_type=" + type_to_str<cost_type>() + ")";
           })
      .def("__eq__",
           [](const self_type &self, const self_type &other) -> auto {
             return self.max_width == other.max_width;
           })
      .def(py::pickle([](const self_type &self)
                          -> auto { return std::tuple{self.max_width}; },
                      [](const std::tuple<width_type> &data) -> auto {
                        return self_type{std::get<0>(data)};
                      }));
}

}  // namespace tnco::optimize::finite_width::cost_model::simple
