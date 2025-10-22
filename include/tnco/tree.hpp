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
#include <stack>
#include <tnco/assert.hpp>
#include <vector>

namespace tnco::tree {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename NodeType>
struct Tree {
  static_assert(std::is_trivially_copyable_v<NodeType>);
  using node_type = NodeType;

  std::vector<node_type> nodes;

  Tree(std::vector<node_type> nodes) : nodes{std::move(nodes)} {
    if (!size()) {
      throw std::invalid_argument("'nodes' cannot be empty.");
    }

    if (const auto [valid_, msg_] = is_valid(); !valid_) {
      throw std::invalid_argument(msg_);
    }
  }

  Tree() : Tree{{node_type{}}} {}

  auto operator==(const Tree &other) const -> bool {
    return nodes == other.nodes;
  }

  [[nodiscard]] auto is_valid() const -> std::pair<bool, std::string> {
    // All node must be valid
    if (!std::all_of(
            std::begin(nodes), std::end(nodes),
            [n_nodes = static_cast<long int>(size())](auto &&node) -> auto {
              for (const auto &x :
                   {node.parent, node.children[0], node.children[1]}) {
                if (!(x == node_type::null || (x >= 0 && x < n_nodes))) {
                  return false;
                }
              }
              return node.is_valid().first;
            })) {
      return {false, "Nodes are not valid"};
    }

    // The last element should be a root
    if (!nodes.back().is_root()) {
      return {false, "Last node should be root."};
    }

    // There should be only one root
    if (std::count_if(std::begin(nodes), std::end(nodes),
                      [](auto &&node) -> auto { return node.is_root(); }) !=
        1) {
      return {false, "There should be only one root."};
    }

    // Get number of leaves
    const auto n_leaves = this->n_leaves();

    // All leaves must appear at the beginning
    if (static_cast<size_t>(std::count_if(
            std::begin(nodes), std::begin(nodes) + n_leaves,
            [](auto &&node) -> auto { return node.is_leaf(); })) != n_leaves) {
      return {false, "All leaves should be first."};
    }

    // Number of nodes is fixed by the number of leaves
    if (size() != (2 * n_leaves) - 1) {
      return {false,
              "Number of nodes is not constenst with the number of leaves."};
    }

    {
      std::vector<size_t> count_parents(size());
      std::vector<size_t> count_children(size());
      std::for_each(std::begin(nodes), std::end(nodes),
                    [&count_parents, &count_children](auto &&node) -> auto {
                      if (!node.is_leaf()) {
                        count_children[node.children[0]] += 1;
                        count_children[node.children[1]] += 1;
                      }
                      if (!node.is_root()) {
                        count_parents[node.parent] += 1;
                      }
                    });

      // No leaves should be the parent of any nodes. All the other nodes must
      // have two children
      if (!std::transform_reduce(std::begin(count_parents),
                                 std::end(count_parents), std::begin(nodes),
                                 true, std::logical_and<bool>{},
                                 [](auto &&c, auto &&node) -> auto {
                                   return c == node.is_leaf() ? 0 : 2;
                                 })) {
        return {false, "Tree is not valid."};
      }

      // All nodes, excluding root, should be the child of a single node
      if (!std::transform_reduce(std::begin(count_children),
                                 std::end(count_children), std::begin(nodes),
                                 true, std::logical_and<bool>{},
                                 [](auto &&c, auto &&node) -> auto {
                                   return c == node.is_root() ? 0 : 1;
                                 })) {
        return {false, "Tree is not valid."};
      }
    }

    // Everything, OK
    return {true, ""};
  }

  void swap_with_nn(const typename node_type::index_type pos_D)
#ifdef NDEBUG
      noexcept
#endif
  {
    /*
     *       A
     *      / \
     *     B   C
     *    / \
     *   E   D
     *
     *   Swap D <-> C.
     */
    if (pos_D >= static_cast<long int>(size())) {
      return;
    }

    // Get node D
    auto &node_D = nodes[pos_D];
    if (node_D.is_root()) {
      return;
    }

    // Get node B
    const auto pos_B = node_D.parent;
    auto &node_B = nodes[pos_B];
    if (node_B.is_root()) {
      return;
    }

    // Get node A
    const auto pos_A = node_B.parent;
    auto &node_A = nodes[pos_A];

    // Get node C
    const auto pos_C = node_A.children[node_A.children[0] == pos_B];
    auto &node_C = nodes[pos_C];

    // Update nodes
    node_A.children[node_A.children[0] != pos_C] = pos_D;
    node_B.children[node_B.children[0] != pos_D] = pos_C;
    node_C.parent = pos_B;
    node_D.parent = pos_A;

    // Check if valid
#ifndef NDEBUG
    if (const auto [valid_, msg_] = is_valid(); !valid_) {
      throw std::invalid_argument(msg_);
    }
#endif
  }

  [[nodiscard]] auto size() const { return std::size(nodes); }

  [[nodiscard]] auto n_leaves() const -> size_t {
    ASSERT(static_cast<size_t>(std::count_if(std::begin(nodes), std::end(nodes),
                                             [](auto &&node) -> auto {
                                               return node.is_leaf();
                                             })) == (size() + 1) / 2,
           "Wrong number of leaves.");
    return (size() + 1) / 2;
  }
};

template <typename... T>
void init(py::module &m, const std::string &name) {
  using self_type = Tree<T...>;

  py::class_<self_type>(m, name.c_str())
      .def(py::init<>())
      .def(py::init<self_type>())
      .def(py::init<decltype(self_type::nodes)>(), "nodes"_a)
      .def("__len__", &self_type::size)
      .def("__eq__", &self_type::operator==)
      .def_property_readonly("n_leaves", &self_type::n_leaves)
      .def_readonly("nodes", &self_type::nodes)
      .def("swap_with_nn", &self_type::swap_with_nn)
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
          "return_message"_a = false)
      .def(py::pickle([](const self_type &self) -> auto { return self.nodes; },
                      [](const decltype(self_type::nodes) &nodes) -> auto {
                        return self_type{nodes};
                      }));
}

}  // namespace tnco::tree
