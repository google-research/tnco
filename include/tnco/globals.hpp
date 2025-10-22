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

#include <random>

#include "bitset.hpp"
#include "ctree.hpp"
#include "node.hpp"
#include "tree.hpp"

#ifndef SKIP_MPFR
#include "fixed_float.hpp"
#endif

namespace tnco {

// Global types
using index_type = int_fast32_t;
using bitset_type = bitset::Bitset<>;
using node_type = node::Node<index_type>;
using tree_type = tree::Tree<node_type>;
using ctree_type = ctree::ContractionTree<tree_type, bitset_type>;
using prob_type = double;
using prng_type = std::mt19937;
using atol_type = double;
using beta_type = double;

// Define higher-precision floats
#ifndef SKIP_MPFR
using float1024 = fixed_float::FixedFloat<1024>;
#endif

// Convert to string
template <typename... T>
auto to_string(const T&... x) -> std::string {
  std::ostringstream ss;
  (ss << ... << x);
  return ss.str();
}

template <typename T>
auto type_to_str() -> std::string {
  using type = std::decay_t<T>;

  // Floating types
  if constexpr (std::is_fundamental_v<type> && std::is_floating_point_v<type>) {
    return "float" + to_string(8 * sizeof(type));
  }

#ifndef SKIP_MPFR
  if constexpr (fixed_float::is_FixedFloat_v<type>) {
    return "float" + to_string(type::prec);
  }
#endif

  // Not implemented
  throw std::invalid_argument("type not supported.");
}

// NOLINTBEGIN(bugprone-macro-parentheses)
#define WRAP(OBJ, METHOD)                                 \
  [&obj = OBJ](auto&&... xs) -> auto {                    \
    return obj.METHOD(std::forward<decltype(xs)>(xs)...); \
  }
// NOLINTEND(bugprone-macro-parentheses)

#ifdef SKIP_MPFR

#define EXPAND_COST_TYPE(function, ...) \
  function<float>(__VA_ARGS__);         \
  function<double>(__VA_ARGS__);        \
  function<long double>(__VA_ARGS__);

#define EXPAND_WIDTH_TYPE(function, T, ...) \
  function<T, float>(__VA_ARGS__);          \
  function<T, double>(__VA_ARGS__);         \
  function<T, long double>(__VA_ARGS__);

#define EXPAND_COST_WIDTH_TYPE(function, ...)       \
  EXPAND_WIDTH_TYPE(function, float, __VA_ARGS__);  \
  EXPAND_WIDTH_TYPE(function, double, __VA_ARGS__); \
  EXPAND_WIDTH_TYPE(function, long double, __VA_ARGS__);

#else

#define EXPAND_COST_TYPE(function, ...) \
  function<float>(__VA_ARGS__);         \
  function<double>(__VA_ARGS__);        \
  function<long double>(__VA_ARGS__);   \
  function<float1024>(__VA_ARGS__);

#define EXPAND_WIDTH_TYPE(function, T, ...) \
  function<T, float>(__VA_ARGS__);          \
  function<T, double>(__VA_ARGS__);         \
  function<T, long double>(__VA_ARGS__);

#define EXPAND_COST_WIDTH_TYPE(function, ...)           \
  EXPAND_WIDTH_TYPE(function, float, __VA_ARGS__);      \
  EXPAND_WIDTH_TYPE(function, double, __VA_ARGS__);     \
  EXPAND_WIDTH_TYPE(function, long double, __VA_ARGS__) \
  EXPAND_WIDTH_TYPE(function, float1024, __VA_ARGS__);

#endif

}  // namespace tnco
