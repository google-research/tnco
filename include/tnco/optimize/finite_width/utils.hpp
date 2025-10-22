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

#include <algorithm>
#include <numeric>
#include <tnco/optimize/infinite_memory/utils.hpp>
#include <tnco/utils.hpp>

namespace tnco::optimize::finite_width::utils {

template <typename CostType, typename CTree, typename CCost, typename Slices>
[[nodiscard]] auto get_cost(const CTree &ctree, const CCost &ccost,
                            const Slices &slices) -> CostType {
  return infinite_memory::utils::get_cost<CostType>(
      ctree, [&ccost, &slices](auto &&...xs) -> auto {
        return ccost(std::forward<decltype(xs)>(xs)..., slices);
      });
}

template <typename CostType>
struct CostCache : infinite_memory::utils::CostCache<CostType> {
  using base_type = infinite_memory::utils::CostCache<CostType>;
  using cost_type = CostType;

  CostCache() = default;

  template <typename CTree, typename CCost, typename Slices>
  CostCache(const CTree &ctree, const CCost &ccost, const Slices &slices)
      : base_type{ctree, [&ccost, &slices](auto &&...x) -> auto {
                    return ccost(std::forward<decltype(x)>(x)..., slices);
                  }} {}
};

template <typename WidthType>
struct WidthCache {
  using width_type = WidthType;

  // Width of each tensor before slicing
  std::vector<width_type> width;

  WidthCache() = default;

  template <typename CTree, typename GetWidth>
  WidthCache(const CTree &ctree, const GetWidth &get_width)
      : width(std::size(ctree)) {
    // Initialize width
    std::transform(std::begin(ctree.inds), std::end(ctree.inds),
                   std::begin(width),
                   [&get_width, &dims = ctree.dims](auto &&xs) -> auto {
                     return get_width(xs, dims);
                   });
  }

  template <typename FloatType>
  auto is_close_to(const WidthCache &cache, const FloatType &atol) const
      -> bool {
    return tnco::utils::all_close(width, cache.width, atol);
  }
};

template <typename FloatType>
struct DimsCache {
  using float_type = FloatType;

  // log2(dimensions)
  std::variant<float_type, std::vector<float_type>> log2_dims;

  DimsCache() = default;

  template <typename CTree>
  DimsCache(const CTree &ctree) {
    // Initialize log2(dims)
    std::visit(
        [&log2_dims = this->log2_dims](auto &&dims) -> auto {
          if constexpr (std::is_arithmetic_v<std::decay_t<decltype(dims)>>) {
            log2_dims = static_cast<float_type>(log2(dims));
          } else {
            std::vector<float_type> log2_dims_(std::size(dims));
            std::transform(std::begin(dims), std::end(dims),
                           std::begin(log2_dims_),
                           [](auto &&d) -> auto { return log2(d); });
            log2_dims = std::move(log2_dims_);
          }
        },
        ctree.dims);
  }

  template <typename FType>
  auto is_close_to(const DimsCache &cache, const FType &atol) const -> bool {
    // Get pointers to vectors
    const auto *const px_ = std::get_if<std::vector<float_type>>(&log2_dims);
    const auto *const py_ =
        std::get_if<std::vector<float_type>>(&cache.log2_dims);

    // If both are vectors
    if (px_ && py_) {
      return tnco::utils::all_close(*px_, *py_, atol);
    }

    // If both are floats
    if (px_ == nullptr && py_ == nullptr) {
      return tnco::utils::is_close(std::get<float_type>(log2_dims),
                                   std::get<float_type>(log2_dims), atol);
    }

    // At this point, they should have different types
    return false;
  }
};

}  // namespace tnco::optimize::finite_width::utils
