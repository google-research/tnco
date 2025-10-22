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
#include <optional>
#include <random>
#include <sstream>
#include <tnco/assert.hpp>
#include <tnco/globals.hpp>
#include <tnco/utils.hpp>
#include <variant>

namespace tnco::optimize::optimizer {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

struct Optimizer {
  using ctree_type = tnco::ctree_type;
  using index_type = typename ctree_type::node_type::index_type;
  using bitset_type = typename ctree_type::bitset_type;
  using prng_type = tnco::prng_type;

  ctree_type ctree;
  const size_t n_tensors{};
  const size_t n_leaves{};
  const bool disable_shared_inds;
  mutable prng_type prng;
  ctree_type min_ctree;

  // Cannot be copied or moved
  Optimizer(const Optimizer &) = delete;
  Optimizer(Optimizer &&) = delete;
  auto operator=(const Optimizer &) -> Optimizer & = delete;
  auto operator=(Optimizer &&) -> Optimizer & = delete;

  Optimizer(ctree_type ctree,
            const std::optional<std::variant<size_t, std::string>> &seed,
            const bool disable_shared_inds,
            std::optional<ctree_type> min_ctree = std::nullopt)
      : ctree{std::move(ctree)},
        n_tensors{std::size(this->ctree)},
        n_leaves{this->ctree.n_leaves()},
        disable_shared_inds{disable_shared_inds},
        min_ctree{min_ctree.has_value() ? std::move(min_ctree.value())
                                        : this->ctree} {
    // Initialize PRNG
    if (seed.has_value()) {
      if (const auto *const p_ = std::get_if<std::string>(&seed.value())) {
        std::istringstream iss{*p_};
        iss >> prng;
      } else {
        prng.seed(std::get<size_t>(seed.value()));
      }
    } else {
      prng.seed(std::random_device()());
    }

    if (const auto [valid_, msg_] = is_valid(); !valid_) {
      throw std::invalid_argument(msg_);
    }
  }

  ~Optimizer() = default;

  [[nodiscard]] auto get_ctree_nn(const index_type pos_B) const
#ifdef NDEBUG
      noexcept
#endif
      -> std::array<index_type, 4> {
    /*
     * Return neighbors of B accordingly to the following schema:
     *
     *      A
     *     / \
     *    B   C
     *   / \
     *  D   E
     *
     * If B is 'null', a root, or a leaf, all returned positions will be
     * 'null'. The order for D and E is randomized. However, if
     * 'this->disable_shared_inds' is 'false, it is guaranteed that D always
     * shares an index with C.
     *
     */
    static constexpr auto null = ctree_type::node_type::null;

    const auto &nodes = ctree.nodes;
    const auto &inds = ctree.inds;

    // If null, just return
    if (pos_B == null) {
      return {null, null, null, null};
    }

    // Collect A, B, C
    const auto &node_B = nodes[pos_B];
    if (node_B.is_root() || node_B.is_leaf()) {
      return {null, null, null, null};
    }
    const auto &pos_A = node_B.parent;
    const auto &node_A = nodes[pos_A];
    const auto &pos_C =
        node_A.children[0] == pos_B ? node_A.children[1] : node_A.children[0];
    const auto &inds_C = inds[pos_C];

    // Collect D and E
    const auto [pos_D, pos_E] =
        [&prng = prng, &disable_shared_inds = disable_shared_inds,
         &pos_0 = node_B.children[0], &pos_1 = node_B.children[1],
         inter_C0 = inds[node_B.children[0]].intersects(inds_C),
         inter_C1 = inds[node_B.children[1]].intersects(inds_C)]() -> auto {
      ASSERT(disable_shared_inds || inter_C0 || inter_C1,
             "Problem with shared inds");
      if (disable_shared_inds || (inter_C0 && inter_C1)) {
        if constexpr (std::is_same_v<std::decay_t<prng_type>, std::nullptr_t>) {
          return std::tuple{pos_0, pos_1};
        } else {
          return prng() % 2 ? std::tuple{pos_0, pos_1}
                            : std::tuple{pos_1, pos_0};
        }
      }
      return inter_C0 ? std::tuple{pos_0, pos_1} : std::tuple{pos_1, pos_0};
    }();

#ifndef NDEBUG
    const auto &node_C = nodes[pos_C];
    const auto &node_D = nodes[pos_D];
    const auto &node_E = nodes[pos_E];
    static constexpr auto is_valid = [](auto &&node, auto &&c0, auto &&c1,
                                        auto &&p) -> auto {
      using c_type = std::array<index_type, 2>;
      if ((c0 != null || c1 != null) && !(node.children == c_type{c0, c1} ||
                                          node.children == c_type{c1, c0})) {
        return false;
      }
      return p == null || node.parent == p;
    };

    ASSERT(pos_A != null && pos_B != null && pos_C != null && pos_D != null &&
               pos_E != null,
           "All nodes should be available.");
    ASSERT(is_valid(node_A, pos_B, pos_C, null) &&
               is_valid(node_B, pos_D, pos_E, pos_A) &&
               is_valid(node_C, null, null, pos_A) &&
               is_valid(node_D, null, null, pos_B) &&
               is_valid(node_E, null, null, pos_B),
           "Failed to get links.");
#endif

    return {pos_A, pos_C, pos_D, pos_E};
  }

  [[nodiscard]] auto is_valid() const -> std::pair<bool, std::string> {
    // Check ctree
    if (const auto [valid_, msg_] = ctree.is_valid(!disable_shared_inds);
        !valid_) {
      return {false, msg_};
    }

    // Check min ctree
    if (const auto [valid_, msg_] = min_ctree.is_valid(!disable_shared_inds);
        !valid_) {
      return {false, msg_};
    }

    // Everything, OK
    return {true, ""};
  }

  [[nodiscard]] auto get_prng_state() const {
    std::ostringstream oss;
    oss << prng;
    return oss.str();
  }
};

void init(py::module &m, const std::string &name) {
  using self_type = Optimizer;
  using ctree_type = typename self_type::ctree_type;
  py::class_<self_type>(m, name.c_str())
      .def(py::init<ctree_type,
                    const std::optional<std::variant<size_t, std::string>> &,
                    const bool, std::optional<ctree_type>>(),
           "ctree"_a, py::kw_only(), "seed"_a = std::nullopt,
           "disable_shared_inds"_a = false, "min_ctree"_a = std::nullopt)
      .def_readonly("ctree", &self_type::ctree)
      .def_readonly("min_ctree", &self_type::min_ctree)
      .def_readonly("disable_shared_inds", &self_type::disable_shared_inds)
      .def_property_readonly("prng_state", &self_type::get_prng_state)
      .def(
          "is_valid",
          [](const self_type &self, const bool return_message)
              -> std::variant<bool, std::pair<bool, std::string>> {
            const auto [valid, msg] = self.is_valid();
            if (return_message) {
              return std::pair{valid, msg};
            }
            return valid;
          },
          py::kw_only(), "return_message"_a = false);
}

}  // namespace tnco::optimize::optimizer
