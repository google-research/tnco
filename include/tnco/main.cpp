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

#include "optimize/main.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bitset.hpp"
#include "ctree.hpp"
#include "globals.hpp"
#include "node.hpp"
#include "tree.hpp"
#include "utils.hpp"

#ifndef SKIP_MPFR
#include "fixed_float.hpp"
#endif

namespace tnco {

// Rename namespace
namespace py = pybind11;

// Initialize main module
PYBIND11_MODULE(tnco_core, m) {
  // Initialize basic types
  bitset::init(m, "Bitset");
  node::init<index_type>(m, "Node");
  tree::init<node_type>(m, "Tree");
  ctree::init<tree_type, bitset_type>(m, "ContractionTree");
#ifndef SKIP_MPFR
  fixed_float::init<1024>(m, "float1024");
#endif

  // Utils
  {
    auto sm = m.def_submodule("utils");
    utils::init(sm);
  }

  {
    auto sm = m.def_submodule("optimize");
    optimize::init(sm);
  }
}

}  // namespace tnco
