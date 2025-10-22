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
#include <numeric>
#include <stack>
#include <tnco/globals.hpp>
#include <vector>

namespace tnco::utils {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename Tree, typename Callback>
auto traverse(const Tree &tree, const Callback &callback) -> void {
  std::stack<size_t> stack;
  std::vector<bool> visited(std::size(tree), false);

  stack.push(std::size(tree) - 1);
  while (std::size(stack)) {
    const auto pos = stack.top();
    if (const auto &node = tree.nodes[pos]; visited[pos] || node.is_leaf()) {
      stack.pop();
      callback(pos);
    } else {
      visited[pos] = true;
      stack.push(node.children[1]);
      stack.push(node.children[0]);
    }
  }
}

template <typename Tree>
auto get_contraction(const Tree &tree)
    -> std::vector<std::array<index_type, 3>> {
  using index_type = typename std::decay_t<Tree>::node_type::index_type;

  // Initialize contraction
  std::vector<std::array<index_type, 3>> contraction;

  // Get contraction
  traverse(tree, [&nodes = tree.nodes, &contraction](auto &&pos) -> auto {
    if (const auto &node = nodes[pos]; !node.is_leaf()) {
      contraction.push_back(
          {node.children[0], node.children[1], static_cast<index_type>(pos)});
    }
  });

  // Return contraction
  return contraction;
}

template <typename ValX, typename ValY, typename Atol>
auto is_close(const ValX &x, const ValY &y, const Atol &atol) -> bool {
  return abs(x - y) <= atol;
}

template <typename ValX, typename ValY, typename Atol>
auto is_logclose(const ValX &x, const ValY &y, const Atol &atol) -> bool {
  if (x < 0 || y < 0) {
    return false;
  }
  if (x == 0 || y == 0) {
    return x == y;
  }
  return abs(log(x) - log(y)) <= atol;
}

template <typename ArrayX, typename ArrayY, typename Compare>
auto compare(const ArrayX &ax, const ArrayY &ay, const Compare &cmp) -> bool {
  if (std::size(ax) != std::size(ay)) {
    return false;
  }
  return std::transform_reduce(
      std::begin(ax), std::end(ax), std::begin(ay), true,
      [](auto &&x, auto &&y) -> auto { return x & y; }, cmp);
}

template <typename ArrayX, typename ArrayY, typename Atol>
auto all_close(const ArrayX &ax, const ArrayY &ay, const Atol &atol) -> bool {
  return compare(ax, ay, [atol](auto &&x, auto &&y) -> auto {
    return is_close(x, y, atol);
  });
}

template <typename ArrayX, typename ArrayY, typename Atol>
auto all_logclose(const ArrayX &ax, const ArrayY &ay, const Atol &atol)
    -> bool {
  return compare(ax, ay, [atol](auto &&x, auto &&y) -> auto {
    return is_logclose(x, y, atol);
  });
}

auto init(py::module &m) -> void {
  m.def("traverse",
        &traverse<const tnco::tree_type &, const std::function<void(size_t)> &>,
        "tree"_a, "callback"_a, py::pos_only());
  m.def("get_contraction", &get_contraction<const tnco::tree_type &>, "tree"_a,
        py::pos_only());
  m.def(
      "is_close",
      [](const long double &x, const long double &y,
         const long double &atol) -> auto { return is_close(x, y, atol); },
      "x"_a, "y"_a, py::pos_only(), py::kw_only(), "atol"_a = 1e-8);
  m.def(
      "is_logclose",
      [](const long double &x, const long double &y,
         const long double &atol) -> auto { return is_logclose(x, y, atol); },
      "x"_a, "y"_a, py::pos_only(), py::kw_only(), "atol"_a = 1e-8);
  m.def(
      "all_close",
      [](const std::vector<long double> &x, const std::vector<long double> &y,
         const long double &atol) -> auto { return all_close(x, y, atol); },
      "x"_a, "y"_a, py::pos_only(), py::kw_only(), "atol"_a = 1e-8);
  m.def(
      "all_logclose",
      [](const std::vector<long double> &x, const std::vector<long double> &y,
         const long double &atol) -> auto { return all_logclose(x, y, atol); },
      "x"_a, "y"_a, py::pos_only(), py::kw_only(), "atol"_a = 1e-8);
}

}  // namespace tnco::utils
