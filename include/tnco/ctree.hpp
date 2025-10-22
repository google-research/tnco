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

#include <numeric>
#include <variant>

namespace tnco::ctree {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename TreeType, typename BitsetType>
struct ContractionTree : TreeType {
  using tree_type = TreeType;
  using bitset_type = BitsetType;
  using dims_int_type = size_t;
  using dims_vec_type = std::vector<size_t>;
  using dims_type = std::variant<dims_int_type, dims_vec_type>;

  std::vector<bitset_type> inds;
  dims_type dims;

  template <typename VectorNodes>
  ContractionTree(VectorNodes &&nodes, std::vector<bitset_type> inds,
                  dims_type dims, const bool check_shared_inds = false)
      : tree_type{std::forward<VectorNodes>(nodes)},
        inds{std::move(inds)},
        dims{std::move(dims)} {
    if (const auto [valid_, msg_] = is_valid(check_shared_inds); !valid_) {
      throw std::invalid_argument(msg_);
    }
  }

  ContractionTree()
      : ContractionTree{std::vector{typename tree_type::node_type{}},
                        {bitset_type{}},
                        dims_int_type{1},
                        false} {}

  template <typename VectorNodes>
  ContractionTree(VectorNodes &&nodes,
                  const std::vector<std::vector<size_t>> &inds, dims_type dims,
                  const bool check_shared_inds = false)
      : tree_type{std::forward<VectorNodes>(nodes)}, dims{std::move(dims)} {
    // Get number of indices
    const auto n_inds = std::transform_reduce(
        std::begin(inds), std::end(inds), size_t{0},
        [](auto &&x, auto &&y) -> auto { return std::max(x, y); },
        [](auto &&xs) -> auto {
          return std::empty(xs)
                     ? 0
                     : *std::max_element(std::begin(xs), std::end(xs)) + 1;
        });

    // Build inds
    std::transform(
        std::begin(inds), std::end(inds), std::back_inserter(this->inds),
        [n_inds](auto &&xs) -> auto { return bitset_type{xs, n_inds}; });

    // Convert to int if possible
    if (const auto *const p_ = std::get_if<dims_vec_type>(&this->dims)) {
      if (const auto &dims_ = *p_;
          std::size(dims_) && std::all_of(std::begin(dims_), std::end(dims_),
                                          [d = dims_[0]](auto &&dim) -> auto {
                                            return dim == d;
                                          })) {
        const auto dim_ = dims_[0];
        this->dims = dim_;
      }
    }

    if (const auto [valid_, msg_] = is_valid(check_shared_inds); !valid_) {
      throw std::invalid_argument(msg_);
    }
  }

  auto operator==(const ContractionTree &other) const -> bool {
    return tree_type::operator==(other) && inds == other.inds &&
           dims == other.dims;
  }

  [[nodiscard]] auto is_valid(const bool check_shared_inds) const
      -> std::pair<bool, std::string> {
    // It must be a valid tree
    if (const auto [valid_, msg_] = tree_type::is_valid(); !valid_) {
      return {false, msg_};
    }

    // Check number of inds
    if (std::size(inds) != this->size()) {
      return {false, "Wrong number of indices."};
    }

    // Check number of inds for each tensor
    if (!std::all_of(std::begin(inds), std::end(inds),
                     [n_inds = n_inds()](auto &&xs) -> auto {
                       return std::size(xs) == n_inds;
                     })) {
      return {false, "Number of indices is not consistent among the tensors."};
    }

    // Check dimensions
    if (const auto *const p_ = std::get_if<dims_vec_type>(&dims)) {
      const auto &dims_ = *p_;
      if (std::size(dims_) != n_inds()) {
        return {false, "Wrong number of dimensions."};
      }
      if (std::any_of(std::begin(dims_), std::end(dims_),
                      [](auto &&d) -> auto { return d == 0; })) {
        return {false, "Dimensions must be positive numbers"};
      }
    } else {
      if (std::get<dims_int_type>(dims) == 0) {
        return {false, "Dimensions must be positive numbers"};
      }
    }

    // Check contraction
    for (std::size_t i_ = 0; i_ < this->size(); ++i_) {
      if (const auto &node_ = this->nodes[i_]; !node_.is_leaf()) {
        if (const auto &xs_c0_ = inds[node_.children[0]],
            &xs_c1_ = inds[node_.children[1]], &xs_ = inds[i_];
            (check_shared_inds && !xs_c0_.intersects(xs_c1_)) ||
            !(xs_c0_ ^ xs_c1_).is_subset_of(xs_) ||
            !xs_.is_subset_of(xs_c0_ | xs_c1_)) {
          return {false, "Contraction is not valid."};
        }
      }
    }

    // Everything, OK
    return {true, ""};
  }

  [[nodiscard]] auto n_inds() const { return std::size(inds[0]); }
};

template <typename... T>
void init(py::module &m, const std::string &name) {
  using self_type = ContractionTree<T...>;
  using tree_type = typename self_type::tree_type;
  py::class_<self_type, tree_type>(m, name.c_str())
      .def(py::init<>())
      .def(py::init<decltype(self_type::nodes), decltype(self_type::inds),
                    decltype(self_type::dims), bool>(),
           "nodes"_a, "inds"_a, "dims"_a, py::kw_only(),
           "check_shared_inds"_a = false)
      .def(py::init<decltype(self_type::nodes),
                    const std::vector<std::vector<size_t>> &,
                    decltype(self_type::dims), bool>(),
           "nodes"_a, "inds"_a, "dims"_a, py::kw_only(),
           "check_shared_inds"_a = false)
      .def(py::init<self_type>())
      .def("__eq__", &self_type::operator==)
      .def_readonly("inds", &self_type::inds)
      .def_readonly("dims", &self_type::dims)
      .def_property_readonly("n_inds", &self_type::n_inds)
      .def(
          "is_valid",
          [](const self_type &self, const bool check_shared_inds,
             const bool return_message)
              -> std::variant<bool, std::pair<bool, std::string>> {
            const auto [valid, msg] = self.is_valid(check_shared_inds);
            if (return_message) {
              return std::pair{valid, msg};
            }
            return valid;
          },
          py::kw_only(), "check_shared_inds"_a = false,
          "return_message"_a = false);
}

}  // namespace tnco::ctree
