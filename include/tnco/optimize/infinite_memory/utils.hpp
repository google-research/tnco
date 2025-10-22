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

#include <numeric>
#include <tnco/utils.hpp>

namespace tnco::optimize::infinite_memory::utils {

template <typename CostType>
struct CostCache {
  using cost_type = CostType;

  std::vector<cost_type> partial_cost;
  std::vector<cost_type> contraction_cost;

  CostCache() = default;

  template <typename CTree, typename CCost>
  CostCache(const CTree &ctree, const CCost &ccost)
      : partial_cost(std::size(ctree)), contraction_cost(std::size(ctree)) {
    tnco::utils::traverse(
        ctree,
        [&ctree, n_inds = ctree.n_inds(), &ccost,
         &cache = *this](auto &&pos) -> auto {
          /*
           *   A <--- This node
           *  / \
           * B   C
           */
          if (const auto &node = ctree.nodes[pos]; node.is_leaf()) {
            cache.contraction_cost[pos] = 0;
            cache.partial_cost[pos] = 0;
          } else {
            const auto &inds_A = ctree.inds[pos];
            const auto &inds_B = ctree.inds[node.children[0]];
            const auto &inds_C = ctree.inds[node.children[1]];
            const auto cost_A = ccost(inds_A, inds_B, inds_C, ctree.dims);
            const auto &pcost_B = cache.partial_cost[node.children[0]];
            const auto &pcost_C = cache.partial_cost[node.children[1]];
            cache.contraction_cost[pos] = cost_A;
            cache.partial_cost[pos] = cost_A + pcost_B + pcost_C;
          }
        });
  }

  template <typename FloatType>
  auto is_close_to(const CostCache &cache, const FloatType &atol) const
      -> bool {
    return tnco::utils::all_logclose(partial_cost, cache.partial_cost, atol) &&
           tnco::utils::all_logclose(contraction_cost, cache.contraction_cost,
                                     atol);
  }
};

template <typename Bitset>
struct HyperCache {
  using bitset_type = Bitset;

  std::vector<bitset_type> hyper_inds;

  HyperCache() = default;

  template <typename CTree>
  HyperCache(const CTree &ctree) {
    const auto n_inds = ctree.n_inds();
    const auto n_tensors = std::size(ctree);
    hyper_inds.resize(n_tensors);

    for (size_t pos = 0; pos < n_tensors; ++pos) {
      if (const auto &node = ctree.nodes[pos]; node.is_leaf()) {
        hyper_inds[pos] = bitset_type(n_inds);
      } else {
        const auto &inds_A = ctree.inds[pos];
        const auto &inds_B = ctree.inds[node.children[0]];
        const auto &inds_C = ctree.inds[node.children[1]];
        hyper_inds[pos] = inds_A & inds_B & inds_C;
      }
    }
  }

  template <typename FloatType>
  auto is_close_to(const HyperCache &cache, const FloatType &atol) const
      -> bool {
    // Check if hyper_inds is close
    return hyper_inds == cache.hyper_inds;
  }
};

template <typename CostType, typename CTree, typename CCost>
[[nodiscard]] auto get_cost(const CTree &ctree, const CCost &ccost)
    -> CostType {
  CostType total_cost{0};
  tnco::utils::traverse(
      ctree, [&ctree, &ccost, &total_cost = total_cost](auto &&pos) -> auto {
        if (const auto &node = ctree.nodes[pos]; !node.is_leaf()) {
          const auto &inds_A = ctree.inds[pos];
          const auto &inds_B = ctree.inds[node.children[0]];
          const auto &inds_C = ctree.inds[node.children[1]];
          total_cost += ccost(inds_A, inds_B, inds_C, ctree.dims);
        }
      });
  return total_cost;
}

}  // namespace tnco::optimize::infinite_memory::utils
