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

#include <array>
#include <cstdint>

namespace tnco::node {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename IndexType>
struct Node {
  // Check IndexType
  static_assert(std::is_arithmetic_v<IndexType> && std::is_signed_v<IndexType>);
  using index_type = IndexType;

  // Define null type
  static constexpr index_type null = -1;

  // Members
  std::array<index_type, 2> children{null, null};
  index_type parent{null};

  Node(std::array<index_type, 2> children, index_type parent)
      : children{std::move(children)}, parent{std::move(parent)} {
    // Make sure that all negative numbers are mapped to 'null'
    this->parent = this->parent < 0 ? null : this->parent;
    this->children[0] = this->children[0] < 0 ? null : this->children[0];
    this->children[1] = this->children[1] < 0 ? null : this->children[1];

    if (const auto [valid_, msg_] = is_valid(); !valid_) {
      throw std::invalid_argument(msg_);
    }
  }

  // Partial constructors
  Node() : Node{{null, null}, null} {}
  Node(index_type parent) : Node{{null, null}, parent} {}
  Node(std::array<index_type, 2> children) : Node{std::move(children), null} {}

  auto operator==(const Node &other) const -> bool {
    return parent == other.parent && children == other.children;
  }

  [[nodiscard]] auto is_root() const -> bool { return parent == null; }

  [[nodiscard]] auto is_leaf() const -> bool { return children[0] == null; }

  [[nodiscard]] auto is_valid() const -> std::pair<bool, std::string> {
    // Both children must have the same sign
    if ((children[0] < 0) ^ (children[1] < 0)) {
      return {false, "Both children must have the same sign."};
    }

    // If negative, it must be 'null'
    for (const auto &x : {children[0], children[1], parent}) {
      if (x < 0 && x != null) {
        return {false, "Node is incosistent."};
      }
    }

    // Check is_leaf
    if (is_leaf() && !(children[0] == null && children[1] == null)) {
      return {false, "'is_leaf()' is not working."};
    }

    // Check is_root
    if (is_root() and parent != null) {
      return {false, "'is_root()' is not working."};
    }

    // Children must be different
    if (!is_leaf() && children[0] == children[1]) {
      return {false, "children must be different."};
    }

    // Parent must be different from children
    if (!is_leaf() && !is_root() &&
        (parent == children[0] || parent == children[1])) {
      return {false, "parent must be different from children."};
    }

    // Everything, OK
    return {true, ""};
  }
};

template <typename... T>
void init(py::module &m, const std::string &name) {
  using self_type = Node<T...>;
  using index_type = typename self_type::index_type;

  static constexpr auto int_to_obj = [](const index_type &x) -> py::object {
    if (x == self_type::null) {
      return py::none();
    }
    return py::cast(x);
  };

  py::class_<self_type>(m, name.c_str())
      .def(py::init<>())
      .def(py::init<decltype(self_type::parent)>(), "parent"_a)
      .def(py::init<decltype(self_type::children)>(), "children"_a)
      .def(py::init<decltype(self_type::children),
                    decltype(self_type::parent)>(),
           "children"_a, "parent"_a)
      .def(py::init<self_type>())
      .def("__eq__", &self_type::operator==)
      .def("__repr__",
           [](const self_type &self) -> auto {
             py::str repr;
             repr += py::str("Node(parent=");
             repr += py::str(int_to_obj(self.parent));
             repr += py::str(", children=");
             repr += py::str(py::make_tuple(int_to_obj(self.children[0]),
                                            int_to_obj(self.children[1])));
             repr += py::str(")");
             return repr;
           })
      .def_property_readonly(
          "parent",
          [](const self_type &self) -> auto { return int_to_obj(self.parent); })
      .def_property_readonly("children",
                             [](const self_type &self) -> auto {
                               return py::make_tuple(
                                   int_to_obj(self.children[0]),
                                   int_to_obj(self.children[1]));
                             })
      .def("is_leaf", &self_type::is_leaf)
      .def("is_root", &self_type::is_root)
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
      .def(py::pickle(
          [](const self_type &self) -> auto {
            return std::tuple{self.children, self.parent};
          },
          [](const std::tuple<decltype(self_type::children),
                              decltype(self_type::parent)> &state) -> auto {
            return self_type{std::get<0>(state), std::get<1>(state)};
          }));
}

}  // namespace tnco::node
