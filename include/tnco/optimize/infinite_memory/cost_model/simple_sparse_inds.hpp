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

#include "simple.hpp"

namespace tnco::optimize::infinite_memory::cost_model::simple_sparse_inds {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename CostType, typename Bitset, typename DimsType>
[[nodiscard]] auto get_cost(const Bitset &inds, const Bitset &sparse_inds,
                            const size_t n_projs, const DimsType &dims)
#ifdef NDEBUG
    noexcept
#endif
    -> CostType {
  static constexpr auto min = [](auto &&x, auto &&y) -> CostType {
    return x < y ? static_cast<CostType>(x) : static_cast<CostType>(y);
  };
  return simple::get_cost<CostType>(inds - sparse_inds, dims) *
         min(simple::get_cost<CostType>(inds & sparse_inds, dims), n_projs);
}

template <typename... T>
struct CostModel : simple::CostModel<T...> {
  using base_type = simple::CostModel<T...>;
  using clone_ptr_type = typename base_type::clone_ptr_type;
  using cost_type = typename base_type::cost_type;
  using bitset_type = typename base_type::bitset_type;
  using dims_type = typename base_type::dims_type;

  bitset_type sparse_inds;
  size_t n_projs;

  CostModel(bitset_type sparse_inds, const size_t n_projs)
      : sparse_inds{std::move(sparse_inds)}, n_projs{n_projs} {
    if (n_projs == 0) {
      throw std::runtime_error("'n_projs' must be a positive number.");
    }
  }

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
        [inds = inds_A | inds_B, &sparse_inds = this->sparse_inds,
         &n_projs = this->n_projs](auto &&dims) -> auto {
          return get_cost<cost_type>(inds, sparse_inds, n_projs, dims);
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
  using bitset_type = typename self_type::bitset_type;
  using cost_type = typename self_type::cost_type;
  py::class_<self_type, base_type>(m, name.c_str())
      .def(py::init<bitset_type, size_t>())
      .def(py::init<self_type>())
      .def("__repr__",
           [](const self_type &self) -> auto {
             return "SimpleCostModelSparseInds(" +
                    std::string(self.sparse_inds) +
                    ", n_projs=" + tnco::to_string(self.n_projs) +
                    ", cost_type=" + type_to_str<cost_type>() + ")";
           })
      .def("__eq__",
           [](const self_type &self, const self_type &other) -> auto {
             return self.n_projs == other.n_projs &&
                    self.sparse_inds == other.sparse_inds;
           })
      .def_readonly("sparse_inds", &self_type::sparse_inds)
      .def_readonly("n_projs", &self_type::n_projs)
      .def(py::pickle(
          [](const self_type &self) -> auto {
            return std::tuple{self.sparse_inds, self.n_projs};
          },
          [](const std::tuple<bitset_type, size_t> &data) -> auto {
            return self_type{std::get<0>(data), std::get<1>(data)};
          }));
}

}  // namespace tnco::optimize::infinite_memory::cost_model::simple_sparse_inds
