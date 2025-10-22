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

#include <tnco/utils.hpp>

namespace tnco::optimize::finite_width::greedy::utils {

template <typename Bitset, typename CTree, typename GetWidth,
          typename GetDeltaWidth, typename MaxWidth, typename PRNG,
          typename Width, typename Log2Dims>
[[nodiscard]] auto get_slices_impl(const CTree &ctree,
                                   const GetWidth &get_width,
                                   const GetDeltaWidth &get_delta_width,
                                   const MaxWidth &max_width,
                                   const std::optional<Bitset> &skip_slices,
                                   PRNG &prng, const Width &width,
                                   const Log2Dims &log2_dims)
#ifdef NDEBUG
    noexcept
#endif
    -> Bitset {

  // Initialize
  Bitset slices(ctree.n_inds());

  // Get the number of times an index appears in a tensor with a width larger
  // than max_width
  std::vector<size_t> n_big_tensors(ctree.n_inds());
  for (size_t tpos = 0, end_ = std::size(ctree); tpos < end_; ++tpos) {
    if (width[tpos] > max_width) {
      ctree.inds[tpos].visit(
          [&n_big_tensors](auto &&pos) -> auto { ++n_big_tensors[pos]; });
    }
  }

  // How to sort
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunused-lambda-capture"
  const auto greater = [&n_big_tensors, &log2_dims](auto &&x,
                                                    auto &&y) -> auto {
    if constexpr (std::is_arithmetic_v<Log2Dims>) {
      return n_big_tensors[x] > n_big_tensors[y];
    } else {
      return n_big_tensors[x] == n_big_tensors[y]
                 ? log2_dims[x] > log2_dims[y]
                 : n_big_tensors[x] > n_big_tensors[y];
    }
  };
#pragma GCC diagnostic pop

  tnco::utils::traverse(ctree, [&](auto &&tpos) -> auto {
    if (width[tpos] > max_width) {
      // Get sliced inds
      auto sliced_xs = ctree.inds[tpos] - slices;

      // Update sliced width
      auto sliced_width = get_width(sliced_xs, ctree.dims);

      // If it's larger than the maximum allowed width ...
      if (sliced_width > max_width) {
        // Get positions
        auto positions =
            (skip_slices.has_value() ? sliced_xs - skip_slices.value()
                                     : sliced_xs)
                .positions();

        // Randomize position (to avoid bias after sorting)
        std::shuffle(std::begin(positions), std::end(positions), prng);

        // Sort
        std::stable_sort(std::begin(positions), std::end(positions), greater);

        // Get new slices
        for (const auto &xpos : positions) {
          ASSERT(!slices.test(xpos), "This index shouldn't be already sliced.");

          // Update slices
          slices.set(xpos);

          // Get the new width
          sliced_width += get_delta_width(sliced_xs, ctree.dims, xpos);

          // Update indices
          sliced_xs.reset(xpos);

          // Break if slices are enough
          if (sliced_width <= max_width) {
            break;
          }
        }
      }
    }
  });

  // Check that all tensors are within max_width
#ifndef NDEBUG
  auto err_msg = std::string();
  std::for_each(
      std::begin(ctree.inds), std::end(ctree.inds),
      [&err_msg, &get_width, &slices, &dims = ctree.dims,
       &max_width](auto &&xs) -> auto {
        if (const auto w = get_width(xs - slices, dims); w > max_width) {
          err_msg += tnco::to_string("Failed to reduce width (expected: '",
                                     max_width, "', got: '", w, "')");
        }
      });
  ASSERT(std::empty(err_msg), err_msg);
#endif

  return slices;
}

template <typename Bitset, typename CTree, typename GetWidth,
          typename GetDeltaWidth, typename MaxWidth, typename PRNG,
          typename Width, typename Log2Dims>
[[nodiscard]] auto get_slices(const CTree &ctree, const GetWidth &get_width,
                              const GetDeltaWidth &get_delta_width,
                              const MaxWidth &max_width,
                              const std::optional<Bitset> &skip_slices,
                              PRNG &&prng, const Width &width,
                              const Log2Dims &log2_dims)
#ifdef NDEBUG
    noexcept
#endif
    -> Bitset {
  return std::visit(
      [&](auto &&dims) -> auto {
        return get_slices_impl<Bitset>(ctree, get_width, get_delta_width,
                                       max_width, skip_slices,
                                       std::forward<PRNG>(prng), width, dims);
      },
      log2_dims);
}

}  // namespace tnco::optimize::finite_width::greedy::utils
