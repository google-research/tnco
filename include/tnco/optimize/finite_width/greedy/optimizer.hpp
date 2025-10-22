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
#include <tnco/optimize/finite_width/utils.hpp>
#include <tnco/optimize/infinite_memory/utils.hpp>
#include <tnco/optimize/optimizer.hpp>
#include <tnco/utils.hpp>
#include <variant>

#include "utils.hpp"

namespace tnco::optimize::finite_width::greedy::optimizer {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename CostModel, typename Probability>
struct Optimizer : optimize::optimizer::Optimizer {
  using base_type = optimize::optimizer::Optimizer;
  using ctree_type = tnco::ctree_type;
  using cmodel_type = CostModel;
  using prob_fn_type = Probability;
  using index_type = typename ctree_type::node_type::index_type;
  using prob_type = tnco::prob_type;
  using cost_type = typename cmodel_type::cost_type;
  using width_type = typename cmodel_type::width_type;
  using bitset_type = typename ctree_type::bitset_type;
  using prng_type = tnco::prng_type;
  using cost_cache_type = finite_width::utils::CostCache<cost_type>;
  using hyper_cache_type = infinite_memory::utils::HyperCache<bitset_type>;
  using width_cache_type = finite_width::utils::WidthCache<width_type>;
  using dims_cache_type = finite_width::utils::DimsCache<width_type>;
  using atol_type = tnco::atol_type;

  std::shared_ptr<cmodel_type> p_cmodel;
  size_t max_number_new_slices{};
  mutable width_cache_type width_cache;
  mutable dims_cache_type dims_cache;
  std::optional<bitset_type> skip_slices;
  bitset_type slices;
  bitset_type min_slices;
  mutable cost_cache_type cost_cache;
  mutable hyper_cache_type hyper_cache;
  cost_type min_total_cost{};

  Optimizer(ctree_type ctree, const cmodel_type &cmodel,
            const size_t &max_number_new_slices,
            std::optional<std::variant<size_t, std::string>> seed,
            const bool disable_shared_inds, const atol_type &atol,
            std::optional<bitset_type> skip_slices = std::nullopt,
            std::optional<ctree_type> min_ctree = std::nullopt,
            std::optional<bitset_type> slices = std::nullopt,
            std::optional<bitset_type> min_slices = std::nullopt)
      : base_type{std::move(ctree), std::move(seed), disable_shared_inds,
                  std::move(min_ctree)},
        p_cmodel{cmodel.clone()},
        max_number_new_slices{max_number_new_slices},
        width_cache{this->ctree, WRAP(*p_cmodel, width)},
        dims_cache{this->ctree},
        skip_slices{std::move(skip_slices)},
        slices{slices.has_value()
                   ? std::move(slices.value())
                   : greedy::utils::get_slices<bitset_type>(
                         this->ctree, WRAP(*p_cmodel, width),
                         WRAP(*p_cmodel, delta_width), p_cmodel->max_width,
                         this->skip_slices, this->prng, width_cache.width,
                         dims_cache.log2_dims)},
        min_slices{min_slices.has_value() ? std::move(min_slices.value())
                                          : this->slices},
        cost_cache{this->ctree, WRAP(*p_cmodel, contraction_cost),
                   this->slices},
        hyper_cache{this->ctree},
        min_total_cost{finite_width::utils::get_cost<cost_type>(
            this->min_ctree, WRAP(*p_cmodel, contraction_cost),
            this->min_slices)} {
    // If the total cost is infinite, there is the indication that the float
    // precision is not enough
    if (const auto log2_tc_ = log2(get_total_cost());
        isinf(log2_tc_) || isnan(log2_tc_)) {
      throw std::domain_error("Precision is too low.");
    }
    if (const auto log2_tc_ = log2(get_min_total_cost());
        isinf(log2_tc_) || isnan(log2_tc_)) {
      throw std::domain_error("Precision is too low.");
    }
    if (const auto [valid_, msg_] = is_valid(atol); !valid_) {
      throw std::invalid_argument(msg_);
    }
  }

  auto update(const prob_fn_type &prob, const bool update_slices = true)
#ifdef NDEBUG
      noexcept
#endif
      -> void {
    static constexpr auto null = ctree_type::node_type::null;
    const auto &cmodel = *p_cmodel;
    auto &nodes = this->ctree.nodes;
    auto &inds = this->ctree.inds;
    const auto &dims = this->ctree.dims;
    auto uniform = std::uniform_real_distribution<prob_type>{};

    // Start by selecting a random leaf
    index_type pos_B = this->prng() % this->n_leaves;
    ASSERT(nodes[pos_B].is_leaf(), "Nodes are in the wrong order.");

    // Get the next node
    if ((pos_B = nodes[pos_B].parent); pos_B == null) {
      return;
    }

    // Initialize total cost
    auto total_cost = get_total_cost();

    // Total cost should be a positive number
    ASSERT(total_cost >= 0, "'total_cost' should always be a positive number.");

    while (true) {
      /*
       *      A
       *     / \
       *    B   C <- Try to swap with E
       *   / \
       *  D   E
       *      ^
       *      |
       *      pos
       */

      // Collect neighbors
      auto [pos_A, pos_C, pos_D, pos_E] = this->get_ctree_nn(pos_B);
      if (pos_A == null) {
        break;
      }

      // Get inds
      const auto &inds_A = inds[pos_A];
      auto &inds_B = inds[pos_B];
      const auto &inds_C = inds[pos_C];
      const auto &inds_D = inds[pos_D];
      const auto &inds_E = inds[pos_E];
      ASSERT(this->disable_shared_inds || inds_D.intersects(inds_C),
             "Problem with shared inds.");

      // Get new inds for B
      auto &hyper_inds_A = hyper_cache.hyper_inds[pos_A];
      auto &hyper_inds_B = hyper_cache.hyper_inds[pos_B];
      auto new_inds_B = (inds_D ^ inds_C) | hyper_inds_A | hyper_inds_B;
      auto &width_B = width_cache.width[pos_B];
      const auto new_width_B = p_cmodel->width(new_inds_B, dims);
      //
      const auto new_sliced_inds_B = new_inds_B - slices;
      auto new_sliced_width_B = p_cmodel->width(new_sliced_inds_B, dims);

      // Get old costs for B and A
      auto &ccost_A = cost_cache.contraction_cost[pos_A];
      auto &ccost_B = cost_cache.contraction_cost[pos_B];

      // If cost propagation must be skipped or not
      bool skip_cost_propagation = false;

      if (new_sliced_width_B <= p_cmodel->max_width) {
        // Get new costs for B and A
        const auto new_ccost_A =
            cmodel.contraction_cost(inds_A, new_inds_B, inds_E, dims, slices);
        const auto new_ccost_B =
            cmodel.contraction_cost(new_inds_B, inds_D, inds_C, dims, slices);

        // Get the delta cost
        const auto delta_cost =
            (new_ccost_B - ccost_B) + (new_ccost_A - ccost_A);

        //-------------------------------

        if (uniform(this->prng) <= prob(delta_cost, total_cost)) {
          // Swap
          this->ctree.swap_with_nn(pos_E);

          // Update inds and hyper-inds
          inds_B = new_inds_B;
          hyper_inds_A = inds_A & inds_B & inds_E;
          hyper_inds_B = inds_B & inds_D & inds_C;

          // Swap nodes
          std::swap(pos_C, pos_E);

          // Update costs
          ccost_B = new_ccost_B;
          ccost_A = new_ccost_A;
          total_cost += delta_cost;

          // Update width
          width_B = new_width_B;

          // Total cost should be a positive number
          ASSERT(total_cost >= 0,
                 "'total_cost' should always be a positive number.");
        }

      } else if (max_number_new_slices > 0) {
        // Initialize new slices
        auto new_slices = slices;

        // Update slices to fit the new tensor
        {
          // Get all positions that are not yet sliced
          auto pos = (skip_slices.has_value()
                          ? new_inds_B - slices - skip_slices.value()
                          : new_inds_B - slices)
                         .positions();
          auto n_pos = std::size(pos);
          size_t n_new_slices = 0;

          // Keep adding new random slices until the sliced width is not larger
          // than max_width
          while (n_new_slices < max_number_new_slices &&
                 new_sliced_width_B > p_cmodel->max_width) {
            // Get a random position
            std::swap(pos[this->prng() % n_pos], pos[n_pos - 1]);

            // Check that new sliced index is not already sliced
            ASSERT(!new_slices.test(pos[n_pos - 1]),
                   "Sliced index shouldn't be already sliced.");

            // Add it to new_slices
            new_slices.set(pos[n_pos - 1]);

            // Update sliced width
            std::visit(
                [&new_sliced_width_B, &pos, &n_pos](auto &&log2_dims) -> auto {
                  if constexpr (std::is_arithmetic_v<
                                    std::decay_t<decltype(log2_dims)>>) {
                    new_sliced_width_B -= log2_dims;
                  } else {
                    new_sliced_width_B -= log2_dims[pos[n_pos - 1]];
                  }
                },
                dims_cache.log2_dims);

            // Update number of available slices
            --n_pos;
            ++n_new_slices;
          }

          // Check number of slices
          ASSERT((new_slices - slices).count() == n_new_slices &&
                     n_new_slices <= max_number_new_slices,
                 "Wrong number of new slices.");

          // Check width
          ASSERT(abs(p_cmodel->width(new_inds_B - new_slices, dims) -
                     new_sliced_width_B) < 1e-5,
                 "Error when getting new slices.");
          ASSERT(n_new_slices >= max_number_new_slices ||
                     new_sliced_width_B <= p_cmodel->max_width,
                 "New slices are not enough to reduce width.");
        }

        if (new_sliced_width_B <= p_cmodel->max_width) {
          // Swap nodes, and recompute the contraction cost from scratch
          std::swap(inds_B, new_inds_B);
          this->ctree.swap_with_nn(pos_E);
          auto new_cost_cache = cost_cache_type{
              this->ctree, WRAP(*p_cmodel, contraction_cost), new_slices};

          // Try to update ...
          if (const auto delta_cost =
                  new_cost_cache.partial_cost.back() - total_cost;
              uniform(this->prng) <= prob(delta_cost, total_cost)) {
            // Swap cost_cache
            cost_cache = std::move(new_cost_cache);

            // Update hyper-inds
            hyper_inds_A = inds_A & inds_B & inds_E;
            hyper_inds_B = inds_B & inds_D & inds_C;

            // Update width
            width_B = new_width_B;

            // Update total cost
            total_cost = get_total_cost();

            // Update slices
            slices = std::move(new_slices);

            // Skip cost propagation
            skip_cost_propagation = true;

          } else {
            // Swap back the nodes
            this->ctree.swap_with_nn(pos_C);
            std::swap(inds_B, new_inds_B);
          }
        }
      }

      // Propagate partial costs
      if (!skip_cost_propagation) {
        cost_cache.partial_cost[pos_B] = cost_cache.partial_cost[pos_D] +
                                         cost_cache.partial_cost[pos_E] +
                                         ccost_B;
        cost_cache.partial_cost[pos_A] = cost_cache.partial_cost[pos_B] +
                                         cost_cache.partial_cost[pos_C] +
                                         ccost_A;
      }

      // Go to the next node
      pos_B = pos_A;
    }

    // At this point, B should be the root
    ASSERT(nodes[pos_B].is_root(), "Last visited node should be a root.");

#ifndef NDEBUG
    if (const auto [valid_, msg_] = this->is_valid(1e-5); !valid_) {
      throw std::logic_error("Something went wrong with the update: " + msg_);
    }

    // Check final cost
    if (const auto rel_error = abs((log(total_cost) - log(get_total_cost())) /
                                   log(get_total_cost()));
        rel_error > 1e-2) {
      throw std::logic_error("Total cost is not properly cached (rel. error: " +
                             tnco::to_string(rel_error * 100) + "%)");
    }
#endif

    // If there are slices ...
    if (update_slices && slices.any()) {
      // Get new slices
      auto new_slices = greedy::utils::get_slices<bitset_type>(
          this->ctree, WRAP(*p_cmodel, width), WRAP(*p_cmodel, delta_width),
          p_cmodel->max_width, skip_slices, this->prng, width_cache.width,
          dims_cache.log2_dims);

      // Get new cost
      auto new_cost_cache = cost_cache_type{
          this->ctree, WRAP(*p_cmodel, contraction_cost), new_slices};

      // If cost improves, keep new slices
      if (new_cost_cache.partial_cost.back() < get_total_cost()) {
        slices = std::move(new_slices);
        cost_cache = std::move(new_cost_cache);
      }
    }

#ifndef NDEBUG
    if (const auto [valid_, msg_] = this->is_valid(1e-5); !valid_) {
      throw std::logic_error("Something went wrong with the update: " + msg_);
    }
#endif

    // Update min
    if (const auto tc_ = get_total_cost(); tc_ < min_total_cost) {
      min_total_cost = tc_;
      this->min_ctree = this->ctree;
      min_slices = slices;
    }
  }

  [[nodiscard]] auto is_valid(const atol_type &atol) const
      -> std::pair<bool, std::string> {
    if (const auto [valid_, msg_] = base_type::is_valid(); !valid_) {
      return {valid_, msg_};
    }

    // Check min ctree cost
    if (!tnco::utils::is_logclose(
            finite_width::utils::get_cost<cost_type>(
                this->min_ctree, WRAP(*p_cmodel, contraction_cost), min_slices),
            min_total_cost, atol)) {
      return {false, "Cost for min ctree is not correct."};
    }

    // All tensors should have a width after slicing smaller than max_width
    if (!std::all_of(std::begin(this->ctree.inds), std::end(this->ctree.inds),
                     [&cmodel = *this->p_cmodel,
                      &max_width = p_cmodel->max_width, &slices = this->slices,
                      &dims = this->ctree.dims](auto &&xs) -> auto {
                       return cmodel.width(xs - slices, dims) <= max_width;
                     })) {
      return {false, "Width larger than allowed width after slicing."};
    }
    if (!std::all_of(
            std::begin(this->min_ctree.inds), std::end(this->min_ctree.inds),
            [&cmodel = *this->p_cmodel, &max_width = p_cmodel->max_width,
             &slices = this->min_slices,
             &dims = this->ctree.dims](auto &&xs) -> auto {
              return cmodel.width(xs - slices, dims) <= max_width;
            })) {
      return {false, "Width larger than allowed width after slicing."};
    }

    // Check CostCache
    if (!cost_cache.is_close_to(
            cost_cache_type{this->ctree, WRAP(*p_cmodel, contraction_cost),
                            slices},
            atol)) {
      return {false, "CostCache is not properly cached."};
    }

    // Check HyperCache
    if (!hyper_cache.is_close_to(hyper_cache_type{this->ctree}, atol)) {
      return {false, "HyperCache is not properly cached."};
    }

    // Check WidthCache
    if (!width_cache.is_close_to(
            width_cache_type{this->ctree, WRAP(*p_cmodel, width)}, atol)) {
      return {false, "WidthCache is not properly cached."};
    }

    // Check DimsCache
    if (!dims_cache.is_close_to(dims_cache_type{this->ctree}, atol)) {
      return {false, "DimsCache is not properly cached."};
    }

    // Everything, OK
    return {true, ""};
  }

  [[nodiscard]] auto get_total_cost() const {
    return cost_cache.partial_cost.back();
  }

  [[nodiscard]] auto get_min_total_cost() const { return min_total_cost; }

  [[nodiscard]] auto cmodel() const -> const cmodel_type & { return *p_cmodel; }
};

template <typename... T>
void init(py::module &m, const std::string &name) {
  using self_type = Optimizer<T...>;
  using base_type = typename self_type::base_type;
  using bitset_type = typename self_type::bitset_type;
  using ctree_type = typename self_type::ctree_type;
  using cmodel_type = typename self_type::cmodel_type;
  using atol_type = typename self_type::atol_type;
  py::class_<self_type, base_type>(m, name.c_str())
      .def(py::init<ctree_type, const cmodel_type &, const size_t &,
                    std::optional<std::variant<size_t, std::string>>,
                    const bool, const atol_type &, std::optional<bitset_type>,
                    std::optional<ctree_type>, std::optional<bitset_type>,
                    std::optional<bitset_type>>(),
           "ctree"_a, "cmodel"_a, py::kw_only(), "max_number_new_slices"_a = 0,
           "seed"_a = std::nullopt, "disable_shared_inds"_a = false,
           "atol"_a = 1e-5, "skip_slices"_a, "min_ctree"_a = std::nullopt,
           "slices"_a = std::nullopt, "min_slices"_a = std::nullopt)
      .def("update", &self_type::update, "prob"_a, py::pos_only(),
           py::kw_only(), "update_slices"_a = true)
      .def_property_readonly("cmodel", &self_type::cmodel)
      .def_property_readonly(
          "total_cost",
          [](const self_type &self) -> auto {
            auto Decimal = py::module_::import("decimal").attr("Decimal");
            return Decimal(tnco::to_string(self.get_total_cost()));
          })
      .def_property_readonly(
          "min_total_cost",
          [](const self_type &self) -> auto {
            auto Decimal = py::module_::import("decimal").attr("Decimal");
            return Decimal(tnco::to_string(self.get_min_total_cost()));
          })
      .def_property_readonly("log2_total_cost",
                             [](const self_type &self) -> auto {
                               return log2(self.get_total_cost());
                             })
      .def_property_readonly("log2_min_total_cost",
                             [](const self_type &self) -> auto {
                               return log2(self.get_min_total_cost());
                             })
      .def_readonly("slices", &self_type::slices)
      .def_readonly("min_slices", &self_type::min_slices)
      .def(
          "is_valid",
          [](const self_type &self, const atol_type &atol,
             const bool return_message)
              -> std::variant<bool, std::pair<bool, std::string>> {
            const auto [valid, msg] = self.is_valid(atol);
            if (return_message) {
              return std::pair{valid, msg};
            }
            return valid;
          },
          py::kw_only(), "atol"_a = 1e-5, "return_message"_a = false);
}

}  // namespace tnco::optimize::finite_width::greedy::optimizer
