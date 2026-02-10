# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools as fts
import itertools as its
import operator as op
import pickle
from collections import Counter, defaultdict
from decimal import Decimal
from math import log2
from os import environ
from random import Random

import more_itertools as mit
import numpy as np
import opt_einsum as oe
import pytest
from quimb.tensor import Tensor, TensorNetwork
from tnco_core import ContractionTree as ContractionTree_

import tnco.utils.tensor as tensor_utils
from tnco.ctree import ContractionTree
from tnco.optimize.finite_width import Optimizer as FW_Optimizer
from tnco.optimize.finite_width.cost_model import \
    SimpleCostModel as FW_SimpleCostModel
from tnco.optimize.infinite_memory import Optimizer as IM_Optimizer
from tnco.optimize.infinite_memory.cost_model import \
    SimpleCostModel as IM_SimpleCostModel
from tnco.optimize.prob import BaseProbability, Greedy, MetropolisHastings
from tnco.ordered_frozenset import OrderedFrozenSet
from tnco.testing.utils import (generate_random_inds, generate_random_tensors,
                                get_connected_components,
                                is_valid_contraction_tree)
from tnco.utils.tensor import \
    decompose_hyper_inds as tensor_decompose_hyper_inds
from tnco.utils.tn import contract
from tnco.utils.tn import decompose_hyper_inds as tn_decompose_hyper_inds
from tnco.utils.tn import (fuse, get_einsum_subscripts,
                           get_random_contraction_path, merge_contraction_paths,
                           read_inds, split_contraction_path)

# Initialize RNG
rng = Random(
    environ.get('PYTEST_SEED') +
    environ.get('PYTEST_XDIST_WORKER') if 'PYTEST_SEED' in environ else None)

# Fix max number of repetitions
max_repeat = max(1, float(environ.get('PYTEST_MAX_REPEAT', 'inf')))

# Fix ratio of number of tests
fraction_n_tests = max(
    min(float(environ.get('PYTEST_FRACTION_N_TESTS', '1')), 1), 0)


def repeat(n: int):
    return pytest.mark.repeat(max(min(n * fraction_n_tests, max_repeat), 1))


@pytest.fixture
def random_seed():
    return rng.randrange(2**32)


@repeat(20)
def test_GenerateRandomTensors(random_seed: int, **kwargs):
    # Initialize RNG
    rng = Random(random_seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(100, 300))
    n_inds = kwargs.get('n_inds', rng.randint(300, 1000))
    k = kwargs.get('k', rng.randint(2, 10))
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, 100))
    n_cc = kwargs.get('n_cc', rng.randint(1, 5))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))

    # Get tensors and the dimension of the inds
    try:
        ts_inds, output_inds = generate_random_tensors(
            n_tensors,
            n_inds,
            k,
            n_output_inds=n_output_inds,
            n_cc=n_cc,
            randomize_names=randomize_names,
            seed=random_seed)
    except ValueError as e:
        if str(e) == "Too few indices.":
            pytest.skip(str(e))
        raise e

    # Get random dimensions
    dims = dict(
        map(lambda x:
            (x, rng.randrange(1, 10)), mit.unique_everseen(
                mit.flatten(ts_inds)))) if rng.randrange(2) else rng.randrange(
                    1, 8)

    # output_inds should be a frozenset
    assert isinstance(output_inds, frozenset)

    # Get hyper-count
    hyper_count = dict(
        its.starmap(lambda x, n: (x, n - 1),
                    Counter(mit.flatten(ts_inds)).items()))

    # All inds with a zero counter must be an output index
    assert frozenset(
        map(op.itemgetter(0),
            filter(lambda x: x[1] == 0,
                   hyper_count.items()))).issubset(output_inds)

    # Get all inds
    all_inds = frozenset(mit.flatten(ts_inds))

    # Check sizes
    assert len(ts_inds) == n_tensors * n_cc
    assert len(output_inds) == n_output_inds * n_cc
    assert len(all_inds) == n_inds * n_cc

    # Check random names
    assert randomize_names or all_inds == frozenset(range(n_inds * n_cc))

    # All inds for each tensor should be different
    assert all(
        map(lambda xs: mit.ilen(mit.unique_everseen(xs)) == len(xs), ts_inds))

    # Check output inds
    assert isinstance(dims, int) or output_inds.issubset(dims)

    # Check that calling twice with the same seed gives the same results
    assert (ts_inds, output_inds) == generate_random_tensors(
        n_tensors,
        n_inds,
        k,
        n_output_inds=n_output_inds,
        n_cc=n_cc,
        randomize_names=randomize_names,
        seed=random_seed)

    # Get connected components
    cc = get_connected_components(ts_inds)

    # Check number of cc
    assert len(cc) == n_cc

    ts_inds = list(map(frozenset, ts_inds))

    # Get inds for the cc
    cc_inds = list(
        map(lambda cc: fts.reduce(op.or_, map(lambda x: ts_inds[x], cc)), cc))

    # No output inds should be shared
    assert all(not cc_inds[i_] & cc_inds[j_]
               for i_ in range(n_cc)
               for j_ in range(i_ + 1, n_cc))

    # Get adj matrix
    adj = tuple(
        map(
            lambda xs: tuple(
                map(op.itemgetter(0),
                    filter(lambda w: xs & w[1], enumerate(ts_inds)))), ts_inds))

    # All element should belong to the same cc
    assert all(
        map(
            lambda cc: frozenset(mit.flatten(map(lambda x: adj[x], cc))) ==
            frozenset(cc), cc))

    def single_cc(cc):
        if not len(cc):
            return True

        # Initialize
        visited = dict(zip(cc, its.repeat(False)))
        stack = [cc[0]]

        # Check visisted
        while stack:
            x = stack.pop()
            visited[x] = True
            stack.extend(filter(lambda x: not visited[x], adj[x]))

        # Return true if all elements have been visited
        return sorted(visited) == sorted(cc) and all(visited.values())

    # Check
    assert all(map(single_cc, cc))


@pytest.mark.timeout(30)
@pytest.mark.usefixtures("timeout")
@repeat(50)
def test_GetRandomContractionPath(random_seed: int, **kwargs):
    # Initialize RNG
    rng = Random(random_seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(100, 300))
    n_inds = kwargs.get('n_inds', rng.randint(300, 1000))
    k = kwargs.get('k', rng.randint(2, 10))
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, 100))
    n_cc = kwargs.get('n_cc', rng.randint(1, 5))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))

    # Generate random tensors
    try:
        ts_inds, output_inds = generate_random_tensors(
            n_tensors,
            n_inds,
            k,
            n_output_inds=n_output_inds,
            n_cc=n_cc,
            randomize_names=randomize_names,
            seed=random_seed)
    except ValueError as e:
        if str(e) == "Too few indices.":
            pytest.skip(str(e))
        raise e

    # Get connected components
    cc = get_connected_components(ts_inds)

    # Get hyper-count
    hyper_count = dict(
        its.starmap(lambda x, n: (x, n - 1),
                    Counter(mit.flatten(ts_inds)).items()))

    # Increment hyper-count for hyper-inds
    for x_ in output_inds:
        hyper_count[x_] += 1

    # Get contraction
    paths = get_random_contraction_path(ts_inds,
                                        seed=random_seed,
                                        merge_paths=False)

    # Calling twice should give the same answer with the same seed
    assert paths == get_random_contraction_path(ts_inds,
                                                seed=random_seed,
                                                merge_paths=False)

    # Check number of connected components
    assert len(paths) == n_cc

    # Copy tensors
    ts_inds_ = [list(map(frozenset, ts_inds)) for _ in range(len(paths))]

    # Paths should have the format [(x, y), ...]
    assert all(map(lambda xs: len(xs) == 2, mit.flatten(paths)))

    # Build contraction
    for path_, ts_ in zip(paths, ts_inds_):
        for x_ in path_:
            x_, y_ = sorted(x_)
            assert 0 <= x_ < y_
            y_ = ts_.pop(y_)
            x_ = ts_.pop(x_)

            # They should share at least one index
            shared_ = x_ & y_
            assert shared_

            # Get new inds
            z_ = x_ ^ y_

            # Update hyper count
            for s_ in shared_:
                assert hyper_count[s_] > 0
                hyper_count[s_] -= 1
                if hyper_count[s_] > 0:
                    z_ |= {s_}

            # Append new inds to tensors
            ts_.append(z_)

    # Only output inds can have the counter set to 1
    assert all(
        its.starmap(lambda x, n: n == (x in output_inds), hyper_count.items()))

    # Remove the original tensors
    ts_inds_ = list(
        map(lambda ts: frozenset(ts).difference(map(frozenset, ts_inds)),
            ts_inds_))

    # Once the original tensors are removed, there should be only one element
    assert all(map(lambda x: len(x) == 1, ts_inds_))

    # Get only the final tensor for each cc
    ts_inds_ = list(map(lambda x: mit.nth(x, 0), ts_inds_))

    # The number of final tensors should be equal to the number of cc
    assert len(ts_inds_) == n_cc

    # No shared inds between cc
    assert all(not ts_inds_[i_] & ts_inds_[j_]
               for i_ in range(n_cc)
               for j_ in range(i_ + 1, n_cc))

    # Get output inds for the cc
    cc_output_inds = list(
        map(
            lambda cc: fts.reduce(
                op.or_, map(lambda x: frozenset(ts_inds[x]), cc)) & output_inds,
            cc))

    # Check one-to-one correspondence of output qubits
    if n_output_inds == 0:
        assert all(
            x_ == y_ == frozenset() for x_, y_ in zip(cc_output_inds, ts_inds_))
    else:
        assert all(
            sum(x_ == y_ for y_ in cc_output_inds) == 1 for x_ in ts_inds_)

    # Get raw contraction
    contractions = get_random_contraction_path(ts_inds,
                                               seed=random_seed,
                                               _return_contraction=True)

    # Check number of connected components
    assert len(contractions) == n_cc

    # Get connected components from contraction
    contr_cc = list(
        map(
            lambda contr: frozenset(
                filter(lambda x: x < len(ts_inds), mit.flatten(contr))),
            contractions))

    # Check one-to-one correspondence
    assert all(
        sum(x_ == y_ for y_ in contr_cc) == 1 for x_ in map(frozenset, cc))


@repeat(20)
def test_GetRandomContractionTree(random_seed: int, **kwargs):
    # Initialize RNG
    rng = Random(random_seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(100, 300))
    n_inds = kwargs.get('n_inds', rng.randint(300, 1000))
    k = kwargs.get('k', rng.randint(2, 10))
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, 100))
    n_cc = kwargs.get('n_cc', rng.randint(1, 5))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))
    verbose = kwargs.get('verbose', False)

    # Get tensors
    try:
        ts_inds, output_inds = generate_random_tensors(
            n_tensors,
            n_inds,
            k,
            n_output_inds=n_output_inds,
            n_cc=n_cc,
            randomize_names=randomize_names,
            seed=random_seed)
    except ValueError as e:
        if str(e) == "Too few indices.":
            pytest.skip(str(e))
        raise e

    # Get connected components from tensors
    cc_from_tensors = get_connected_components(ts_inds, verbose=verbose)

    # Get random dimensions
    dims = dict(
        map(lambda x:
            (x, rng.randrange(1, 10)), mit.unique_everseen(
                mit.flatten(ts_inds)))) if rng.randrange(2) else rng.randrange(
                    1, 8)

    # Get all available inds
    all_inds = frozenset(mit.flatten(ts_inds))

    # Get hyper-count
    hyper_count = dict(
        its.starmap(lambda x, n: (x, n - 1),
                    Counter(mit.flatten(ts_inds)).items()))

    # Add one count for output inds
    for x_ in output_inds:
        hyper_count[x_] += 1

    # Get contraction
    paths = get_random_contraction_path(ts_inds,
                                        seed=random_seed,
                                        merge_paths=False)

    # Check number of connected components
    assert len(paths) == n_cc

    # Build cc from paths
    cc = []
    for path_ in paths:

        contraction_ = []
        pos_ = list(range(len(ts_inds)))
        for i_, xs_ in enumerate(path_):
            x_, y_ = sorted(xs_)
            py_ = pos_.pop(y_)
            px_ = pos_.pop(x_)
            pos_.append(i_ + len(ts_inds))
            contraction_.append((px_, py_, pos_[-1]))

        cc.append(
            tuple(
                sorted(
                    filter(lambda x: x < len(ts_inds),
                           mit.unique_everseen(mit.flatten(contraction_))))))

    # Cross check
    assert all(sum(x_ == y_ for y_ in cc_from_tensors) == 1 for x_ in cc)

    # Get contraction trees
    ctrees = [
        ContractionTree(path_,
                        ts_inds,
                        dims,
                        output_inds=output_inds,
                        check_shared_inds=True) for path_ in paths
    ]

    # Check cache
    assert all(map(lambda ctree: ctree._n_tensors == len(ts_inds), ctrees))
    assert all(map(lambda ctree, cc: ctree._tensors_pos == cc, ctrees, cc))
    assert all(
        map(
            lambda ctree, inds: frozenset(ctree._inds_order) == inds, ctrees,
            map(
                lambda cc: frozenset(mit.flatten(map(lambda x: ts_inds[x], cc))
                                    ), cc)))

    # Check inds
    assert all(ctree.all_inds() == frozenset(
        mit.flatten(map(lambda x: ts_inds[x], cc)))
               for ctree, cc in zip(ctrees, cc))

    # Check dimensions of core
    assert all(
        map(
            lambda ctree: type(ContractionTree_(ctree).dims) is
            (int if isinstance(dims, int) else list), ctrees))
    assert all(
        map(lambda ctree: ContractionTree_(ctree).dims == dims,
            ctrees)) if isinstance(dims, int) else all(
                map(
                    lambda ctree: list(map(ctree.dims.get, ctree._inds_order))
                    == ContractionTree_(ctree).dims, ctrees))

    # Calling twice should give the same answer
    assert ctrees == [
        ContractionTree(path_,
                        ts_inds,
                        dims,
                        output_inds=output_inds,
                        check_shared_inds=True) for path_ in paths
    ]

    # Check number of connected components
    assert len(ctrees) == n_cc

    # Check if contraction trees are valid
    for ctree_, cc_ in zip(ctrees, cc):
        ts_inds_ = tuple(map(lambda x: ts_inds[x], cc_))
        all_inds_ = frozenset(mit.flatten(ts_inds_))
        output_inds_ = output_inds & all_inds_
        assert is_valid_contraction_tree(ctree_,
                                         ts_inds=ts_inds_,
                                         dims=dims,
                                         output_inds=output_inds_,
                                         hyper_count=hyper_count,
                                         check_shared_inds=True)

    # Get output inds
    output_inds_from_ct = tuple(map(lambda ct: ct.output_inds(), ctrees))

    # No output inds shared between the cc
    assert all(not output_inds_from_ct[i_] & output_inds_from_ct[j_]
               for i_ in range(n_cc)
               for j_ in range(i_ + 1, n_cc))

    # Check output inds
    assert fts.reduce(op.or_, output_inds_from_ct) == output_inds

    # Check correspondence of output qubits
    assert all(
        its.starmap(
            lambda cc, output_xs: fts.reduce(
                op.or_, map(lambda p: frozenset(ts_inds[p]), cc)) & output_inds
            == output_xs, zip(cc, output_inds_from_ct)))

    # Get all available indices from ctrees
    all_inds_from_ct = tuple(map(lambda ct: ct.all_inds(), ctrees))

    # Inds shouldn't intersect
    assert all(not all_inds_from_ct[i_] & all_inds_from_ct[j_]
               for i_ in range(n_cc)
               for j_ in range(i_ + 1, n_cc))

    # Compare against all inds
    assert fts.reduce(op.or_, all_inds_from_ct) == all_inds

    # Count how many times an index is contracted
    hyper_count_from_ct = defaultdict(int)

    def count(ctree):
        for node_ in ctree.nodes:
            if not node_.is_leaf():
                ix_, iy_ = ctree.inds[node_.children[0]], ctree.inds[
                    node_.children[1]]
                assert ix_ & iy_

                for is_ in ix_ & iy_:
                    hyper_count_from_ct[is_] += 1

    # Count hyper-indices
    mit.consume(map(count, ctrees))

    # Increment by one for all the output inds
    for x_ in mit.flatten(output_inds_from_ct):
        hyper_count_from_ct[x_] += 1

    # Check with the original hyper count
    assert hyper_count_from_ct == hyper_count

    def get_contraction(ctree):
        # Initialize contraction
        contraction = []

        # Get contraction
        pos_ = list(range(len(ts_inds)))
        for i_, xs_ in enumerate(ctree.path()):
            x_, y_ = sorted(xs_)
            py_ = pos_.pop(y_)
            px_ = pos_.pop(x_)
            pos_.append(i_ + len(ts_inds))
            contraction.append((px_, py_, pos_[-1]))

        return contraction

    # Check paths
    assert all(
        map(
            lambda ctree, cc: tuple(
                sorted(
                    filter(
                        lambda x: x < len(ts_inds),
                        mit.unique_everseen(mit.flatten(get_contraction(ctree)))
                    ))) == cc, ctrees, cc))

    # Check contraction tree built from path
    assert all(
        map(
            lambda ctree: ContractionTree(ctree.path(),
                                          ts_inds,
                                          dims,
                                          output_inds=output_inds,
                                          check_shared_inds=True).path() ==
            ctree.path(), ctrees))


@pytest.mark.timeout(30)
@pytest.mark.usefixtures("timeout")
@repeat(30)
def test_OptimizerInfiniteMemory(random_seed: int, **kwargs):

    def log2(x):
        return float(Decimal(x).log10() / Decimal(2).log10())

    # Initialize RNG
    rng = Random(random_seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(50, 100))
    n_inds = kwargs.get('n_inds', rng.randint(100, 200))
    k = kwargs.get('k', rng.randint(2, 10))
    n_cc = 1
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, 100))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))
    disable_shared_inds = kwargs.get('disable_shared_inds',
                                     rng.choice([True, False]))
    atol = kwargs.get('atol', 1e-5)
    cost_type = kwargs.get('cost_type',
                           rng.choice(['float64', 'float128', 'float1024']))

    # Get cost given contraction tree
    def PySimpleContractionCost(ctree):

        cost = 0
        for pos in range(len(ctree)):
            if not (node := ctree.nodes[pos]).is_leaf():
                inds_p = ctree.inds[pos]
                inds_c0 = ctree.inds[node.children[0]]
                inds_c1 = ctree.inds[node.children[1]]
                cost += fts.reduce(
                    op.mul, map(ctree.dims.get, (inds_p | inds_c0 | inds_c1)),
                    1)

        return cost

    # Get tensors
    try:
        ts_inds, output_inds = generate_random_tensors(
            n_tensors,
            n_inds,
            k,
            n_output_inds=n_output_inds,
            n_cc=n_cc,
            randomize_names=randomize_names,
            seed=random_seed)
    except ValueError as e:
        if str(e) == "Too few indices.":
            pytest.skip(str(e))
        raise e

    # Get random dimensions
    dims = dict(
        map(lambda x:
            (x, rng.randrange(1, 10)), mit.unique_everseen(
                mit.flatten(ts_inds)))) if rng.randrange(2) else rng.randrange(
                    1, 8)

    # Get contraction
    paths = get_random_contraction_path(ts_inds,
                                        seed=random_seed,
                                        merge_paths=False)

    # There should be only one cc
    assert len(paths) == 1

    # Get random contraction trees
    ctree = ContractionTree(paths[0],
                            ts_inds,
                            dims,
                            output_inds=output_inds,
                            check_shared_inds=True)

    # Get indices for leaves
    leaf_inds = ctree.inds[:ctree.n_leaves]

    # Get optimizers
    opt = IM_Optimizer(ctree,
                       IM_SimpleCostModel(cost_type=cost_type),
                       seed=random_seed,
                       disable_shared_inds=disable_shared_inds)

    # Check type
    assert opt.cmodel.cost_type == cost_type
    assert pickle.loads(pickle.dumps(opt)).cmodel.cost_type == cost_type

    # Check min
    assert opt.min_ctree == opt.ctree == ctree
    assert abs(log2(opt.min_total_cost) - log2(opt.total_cost)) < atol
    assert abs(log2(opt.total_cost) - opt.log2_total_cost) < atol
    assert abs(log2(opt.min_total_cost) - opt.log2_min_total_cost) < atol
    assert abs(log2(opt.total_cost) - log2(opt.min_total_cost)) < atol
    assert abs(log2(opt.total_cost) -
               log2(PySimpleContractionCost(ctree))) < atol

    # Calling twice should give the same
    assert opt == IM_Optimizer(ctree,
                               IM_SimpleCostModel(cost_type=cost_type),
                               seed=random_seed,
                               disable_shared_inds=disable_shared_inds)

    # Greedy optimization
    greedy = Greedy(cost_type=cost_type)
    total_cost = opt.total_cost
    for i_ in range(100):
        opt.update(greedy)
        assert opt.is_valid() and log2(opt.total_cost) - log2(total_cost) < atol
        # This may fail because greedy updates even when delta_cost == 0
        # assert opt.ctree == opt.min_ctree
        assert abs(log2(opt.min_total_cost) - log2(opt.total_cost)) < atol
        assert abs(log2(opt.total_cost) - opt.log2_total_cost) < atol
        assert abs(log2(opt.min_total_cost) - opt.log2_min_total_cost) < atol
        total_cost = opt.total_cost
        if (i_ % 10) == 0:
            assert abs(
                log2(opt.total_cost) -
                log2(PySimpleContractionCost(opt.ctree))) < atol
            assert abs(
                log2(opt.total_cost) -
                log2(PySimpleContractionCost(opt.min_ctree))) < atol

    # Inds for leaves shouldn't change
    assert opt.ctree.inds[:opt.ctree.n_leaves] == leaf_inds
    assert opt.min_ctree.inds[:opt.min_ctree.n_leaves] == leaf_inds

    # Make a copy
    opt_copy = pickle.loads(pickle.dumps(opt))

    # Check equality
    # This could fail because rounding errors in floating values could lead
    # to two different min ctree
    # assert opt == opt_copy
    assert opt.ctree == opt_copy.ctree
    assert opt.cmodel == opt_copy.cmodel
    assert opt.prng_state == opt_copy.prng_state
    assert opt.disable_shared_inds == opt_copy.disable_shared_inds

    # Optimize (mh)
    mh = MetropolisHastings(cost_type=cost_type)
    for beta_ in range(100):
        mh.beta = beta_
        opt.update(mh)
        opt_copy.update(mh)
        assert opt.is_valid() and opt_copy.is_valid()
        assert opt.ctree == opt_copy.ctree
        # This could fail because rounding errors in floating values could lead
        # to two different min ctree
        # assert opt.min_ctree == opt_copy.min_ctree
        assert abs(log2(opt.total_cost) - log2(opt_copy.total_cost)) < atol
        assert abs(log2(opt.min_total_cost) -
                   log2(opt_copy.min_total_cost)) < atol
        assert abs(log2(opt.total_cost) - opt.log2_total_cost) < atol
        assert abs(log2(opt.min_total_cost) - opt.log2_min_total_cost) < atol
        assert opt.min_total_cost <= opt.total_cost
        if (beta_ % 10) == 0:
            assert abs(
                log2(opt.total_cost) -
                log2(PySimpleContractionCost(opt.ctree))) < atol
            assert abs(
                log2(opt.min_total_cost) -
                log2(PySimpleContractionCost(opt.min_ctree))) < atol

    # Inds for leaves shouldn't change
    assert opt.ctree.inds[:opt.ctree.n_leaves] == leaf_inds
    assert opt.min_ctree.inds[:opt.min_ctree.n_leaves] == leaf_inds
    assert opt_copy.ctree.inds[:opt.ctree.n_leaves] == leaf_inds
    assert opt_copy.min_ctree.inds[:opt.min_ctree.n_leaves] == leaf_inds

    # Check equality
    assert opt == opt_copy

    # Optimize only original
    for i_ in range(100):
        opt.update(BaseProbability(cost_type=cost_type))
        assert abs(log2(opt.total_cost) - opt.log2_total_cost) < atol
        assert abs(log2(opt.min_total_cost) - opt.log2_min_total_cost) < atol
        assert opt.is_valid()
        if (i_ % 10) == 0:
            assert abs(
                log2(opt.total_cost) -
                log2(PySimpleContractionCost(opt.ctree))) < atol
            assert abs(
                log2(opt.min_total_cost) -
                log2(PySimpleContractionCost(opt.min_ctree))) < atol

    # Inds for leaves shouldn't change
    assert opt.ctree.inds[:opt.ctree.n_leaves] == leaf_inds
    assert opt.min_ctree.inds[:opt.min_ctree.n_leaves] == leaf_inds

    # It should differ from copy
    assert opt != opt_copy


@pytest.mark.timeout(30)
@pytest.mark.usefixtures("timeout")
@repeat(30)
def test_OptimizerFiniteWidth(random_seed: int, **kwargs):

    def log2(x):
        return float(Decimal(x).log10() / Decimal(2).log10())

    # Initialize RNG
    rng = Random(random_seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(50, 100))
    n_inds = kwargs.get('n_inds', rng.randint(100, 200))
    k = kwargs.get('k', rng.randint(2, 10))
    n_cc = 1
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, 100))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))
    disable_shared_inds = kwargs.get('disable_shared_inds',
                                     rng.choice([True, False]))
    atol = kwargs.get('atol', 1e-5)
    cost_type = kwargs.get('cost_type',
                           rng.choice(['float64', 'float128', 'float1024']))
    width_type = kwargs.get('width_type', rng.choice(['float64', 'float128']))

    # Get cost given contraction tree
    def PySimpleContractionCost(ctree, slices):

        cost = 0
        for pos in range(len(ctree)):
            if not (node := ctree.nodes[pos]).is_leaf():
                inds_p = ctree.inds[pos]
                inds_c0 = ctree.inds[node.children[0]]
                inds_c1 = ctree.inds[node.children[1]]
                cost += fts.reduce(
                    op.mul,
                    map(ctree.dims.get, (inds_p | inds_c0 | inds_c1 | slices)),
                    1)

        return cost

    def get_width(inds, dims):
        if isinstance(dims, int):
            return log2(dims) * len(inds)
        return sum(map(lambda x: log2(dims[x]), inds))

    # Get tensors
    try:
        ts_inds, output_inds = generate_random_tensors(
            n_tensors,
            n_inds,
            k,
            n_output_inds=n_output_inds,
            n_cc=n_cc,
            randomize_names=randomize_names,
            seed=random_seed)
    except ValueError as e:
        if str(e) == "Too few indices.":
            pytest.skip(str(e))
        raise e

    # Get random dimensions
    dims = dict(
        map(lambda x:
            (x, rng.randrange(1, 10)), mit.unique_everseen(
                mit.flatten(ts_inds)))) if rng.randrange(2) else rng.randrange(
                    1, 8)

    # Get contraction
    paths = get_random_contraction_path(ts_inds,
                                        seed=random_seed,
                                        merge_paths=False)

    # There should be only one cc
    assert len(paths) == 1

    # Get random contraction trees
    ctree = ContractionTree(paths[0],
                            ts_inds,
                            dims,
                            output_inds=output_inds,
                            check_shared_inds=True)

    # Get indices for leaves
    leaf_inds = ctree.inds[:ctree.n_leaves]

    # Fix max width
    max_width = max(map(lambda xs: get_width(xs, dims),
                        ctree.inds)) * rng.random()

    # Get optimizers
    opt = FW_Optimizer(ctree,
                       FW_SimpleCostModel(max_width=max_width,
                                          width_type=width_type,
                                          cost_type=cost_type),
                       seed=random_seed,
                       disable_shared_inds=disable_shared_inds)

    # Check type
    assert opt.cmodel.cost_type == cost_type
    assert opt.cmodel.width_type == width_type
    assert pickle.loads(pickle.dumps(opt)).cmodel.cost_type == cost_type
    assert pickle.loads(pickle.dumps(opt)).cmodel.width_type == width_type

    # Check min
    assert opt.min_ctree == opt.ctree == ctree
    assert abs(log2(opt.min_total_cost) - log2(opt.total_cost)) < atol
    assert abs(log2(opt.total_cost) - opt.log2_total_cost) < atol
    assert abs(log2(opt.min_total_cost) - opt.log2_min_total_cost) < atol
    assert abs(log2(opt.total_cost) - log2(opt.min_total_cost)) < atol
    assert abs(
        log2(opt.total_cost) -
        log2(PySimpleContractionCost(ctree, opt.slices))) < atol

    # Calling twice should give the same
    assert opt == FW_Optimizer(ctree,
                               FW_SimpleCostModel(max_width=max_width,
                                                  cost_type=cost_type,
                                                  width_type=width_type),
                               seed=random_seed,
                               disable_shared_inds=disable_shared_inds)

    # Greedy optimization
    greedy = Greedy(cost_type=cost_type)
    total_cost = opt.total_cost
    for i_ in range(100):
        opt.update(greedy, update_slices=(i_ % 10 == 0))
        assert opt.is_valid() and log2(opt.total_cost) - log2(total_cost) < atol
        # This may fail because greedy updates even when delta_cost == 0
        # assert opt.ctree == opt.min_ctree
        assert abs(log2(opt.min_total_cost) - log2(opt.total_cost)) < atol
        assert abs(log2(opt.total_cost) - opt.log2_total_cost) < atol
        assert abs(log2(opt.min_total_cost) - opt.log2_min_total_cost) < atol
        total_cost = opt.total_cost
        if (i_ % 10) == 0:
            assert abs(
                log2(opt.total_cost) -
                log2(PySimpleContractionCost(opt.ctree, opt.slices))) < atol
            assert abs(
                log2(opt.total_cost) -
                log2(PySimpleContractionCost(opt.min_ctree, opt.min_slices))
            ) < atol

    # Inds for leaves shouldn't change
    assert opt.ctree.inds[:opt.ctree.n_leaves] == leaf_inds
    assert opt.min_ctree.inds[:opt.min_ctree.n_leaves] == leaf_inds

    # Make a copy
    opt_copy = pickle.loads(pickle.dumps(opt))

    # Check equality
    assert opt == opt_copy

    # Optimize (mh)
    mh = MetropolisHastings(cost_type=cost_type)
    for beta_ in range(100):
        mh.beta = beta_
        opt.update(mh, update_slices=(beta_ % 10 == 0))
        assert opt.is_valid()
        # This could fail because rounding errors in floating values could lead
        # to two different min ctree
        assert abs(log2(opt.total_cost) - opt.log2_total_cost) < atol
        assert abs(log2(opt.min_total_cost) - opt.log2_min_total_cost) < atol
        assert opt.min_total_cost <= opt.total_cost
        if (beta_ % 10) == 0:
            assert abs(
                log2(opt.total_cost) -
                log2(PySimpleContractionCost(opt.ctree, opt.slices))) < atol
            assert abs(
                log2(opt.min_total_cost) -
                log2(PySimpleContractionCost(opt.min_ctree, opt.min_slices))
            ) < atol

    # Inds for leaves shouldn't change
    assert opt.ctree.inds[:opt.ctree.n_leaves] == leaf_inds
    assert opt.min_ctree.inds[:opt.min_ctree.n_leaves] == leaf_inds


@repeat(20)
def test_ReadInds(random_seed: int, **kwargs):
    # Initialize RNG
    rng = Random(random_seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(100, 300))
    n_inds = kwargs.get('n_inds', rng.randint(300, 1000))
    k = kwargs.get('k', rng.randint(2, 10))
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, 100))
    n_sparse_inds = kwargs.get('n_output_inds', rng.randint(0, 100))
    n_cc = kwargs.get('n_cc', rng.randint(1, 5))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))
    output_index_token = generate_random_inds(1, seed=rng.randrange(2**32))[0]
    sparse_index_token = generate_random_inds(1, seed=rng.randrange(2**32))[0]

    # Get tensors and the dimension of the inds
    try:
        ts_inds, output_inds = generate_random_tensors(
            n_tensors,
            n_inds,
            k,
            n_output_inds=n_output_inds,
            n_cc=n_cc,
            randomize_names=randomize_names,
            seed=random_seed)
    except ValueError as e:
        if str(e) == "Too few indices.":
            pytest.skip(str(e))
        raise e

    # Get all inds
    all_inds = frozenset(mit.flatten(ts_inds))

    # Get sparse inds
    sparse_inds = frozenset(rng.sample(list(all_inds), k=n_sparse_inds))

    # Get random dimensions
    dims = dict(zip(all_inds, rng.choices(range(100), k=len(all_inds))))

    # Get random names for tensors
    tensor_names = generate_random_inds(len(ts_inds), seed=rng.randrange(2**32))

    # Get inds
    inds = defaultdict(set)
    for name_, xs_ in zip(tensor_names, ts_inds):
        for x_ in xs_:
            inds[x_] |= {name_}

    # Add output index token
    for x_ in output_inds:
        inds[x_] |= {output_index_token}

    # Add sparse inds
    for x_ in sparse_inds:
        inds[x_] |= {sparse_index_token}

    # Add dimension
    inds = dict(
        its.starmap(lambda x, ts: (x, (dims[x],) + tuple(ts)), inds.items()))

    # Get tensors from indices
    tensor_map, dims_, output_inds_, sparse_inds_ = read_inds(
        inds,
        output_index_token=output_index_token,
        sparse_index_token=sparse_index_token)

    for name_, xs_ in tensor_map.items():
        # Check inds
        assert frozenset(ts_inds[tensor_names.index(name_)]) == frozenset(xs_)

    # Check output / sparse inds
    assert output_inds == frozenset(output_inds_)
    assert sparse_inds == frozenset(sparse_inds_)

    # Check dimensions
    assert dims == dims_


@repeat(20)
def test_DecomposeHyperInds(random_seed):
    # Initialize random generator
    rng = Random(random_seed)
    np_rng = np.random.default_rng(seed=random_seed)

    # How to permute
    def permutation(x):
        x = list(x)
        return rng.sample(x, k=len(x))

    # Number of indices for the hyper-index
    k = rng.randrange(3, 7)

    # Dimension of the hyper-index
    dim_hyper = rng.randrange(3, 9)

    # Rest of the dimensions
    p_shape = tuple(rng.randrange(3, 9) for _ in range(rng.randrange(3, 6)))

    # Initialize array
    array = np.zeros((dim_hyper,) * k + p_shape)

    # Fill it, keeping the first k indices for the hyper-inds
    array[(range(dim_hyper),) * k] = np_rng.normal(size=(dim_hyper,) + p_shape)

    # Generate random names
    inds = generate_random_inds(array.ndim)

    # Random permutation
    array = Tensor(array, inds)
    array = array.transpose(*permutation(inds))

    # Get decomposition
    (d_array,
     d_inds), merged_inds = tensor_decompose_hyper_inds(array.data, array.inds)

    # Check number of inds associated to hyper-inds
    assert len(merged_inds) == 1 and len(mit.nth(merged_inds.values(),
                                                 0)) == (k - 1)

    # Check that the axes have been correctly identified
    assert {mit.nth(merged_inds, 0)} | mit.nth(merged_inds.values(),
                                               0) == frozenset(inds[:k])

    # Build diagonal tensor
    d_array_ex = array.transpose(*inds).data
    d_array_ex = Tensor([d_array_ex[(i_,) * k] for i_ in range(dim_hyper)],
                        (mit.nth(merged_inds, 0),) +
                        inds[k:]).transpose(*d_inds)

    # Check if diagonal corresponds
    np.testing.assert_allclose((d_array_ex - d_array).data, 0, atol=1e-5)


@repeat(20)
def test_TensorGetEinsumSubscripts(random_seed):
    # Initialize random generator
    rng = Random(random_seed)
    np_rng = np.random.default_rng(seed=random_seed)

    # How to permute
    def permutation(x):
        x = list(x)
        return rng.sample(x, k=len(x))

    # Get number of inds
    n_inds_a = rng.randrange(3, 8)
    n_inds_b = rng.randrange(3, 8)
    n_shared_inds = rng.randrange(1, min((4, n_inds_a + 1, n_inds_b + 1)))

    # Get random names
    inds_a = generate_random_inds(n_inds_a)
    inds_b = inds_a[:n_shared_inds] + generate_random_inds(n_inds_b -
                                                           n_shared_inds)

    # Randomly permute them
    inds_a, inds_b = map(tuple, map(lambda x: permutation(x), (inds_a, inds_b)))

    # Get random dimensions
    dims = dict(
        map(lambda x: (x, rng.randrange(1, 5)),
            mit.unique_everseen(inds_a + inds_b)))
    dims_a = tuple(map(dims.get, inds_a))
    dims_b = tuple(map(dims.get, inds_b))

    # Generate array
    array_a = np_rng.normal(size=dims_a)
    array_b = np_rng.normal(size=dims_b)

    # Get random output inds
    output_inds = tuple(frozenset(inds_a).symmetric_difference(inds_b))
    output_inds = output_inds + tuple(
        map(lambda x: inds_a[x], rng.sample(range(len(inds_a)), k=2)))
    output_inds = output_inds + tuple(
        map(lambda x: inds_b[x], rng.sample(range(len(inds_b)), k=2)))
    output_inds = tuple(mit.unique_everseen(output_inds))
    output_inds = permutation(output_inds)

    # Compute tensordot providing the output inds
    array_c = np.einsum(
        tensor_utils.get_einsum_subscripts(inds_a, inds_b, output_inds),
        array_a, array_b)

    # Check if correct
    np.testing.assert_allclose(
        (Tensor(array_a, inds_a) &
         Tensor(array_b, inds_b)).contract(output_inds=output_inds).data,
        array_c,
        atol=1e-5)


@repeat(20)
def test_TNGetEinsumSubscripts(random_seed: int, **kwargs):
    # Initialize RNG
    rng = Random(random_seed)

    # Initialize numpy rng
    np_rng = np.random.default_rng(random_seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(5, 15))
    n_inds = kwargs.get('n_inds', rng.randint(25, 45))
    k = kwargs.get('k', rng.randint(2, 6))
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, 10))
    n_cc = kwargs.get('n_cc', rng.randint(1, 3))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))

    # Get random tensors
    ts_inds, output_inds = generate_random_tensors(
        n_tensors=n_tensors,
        n_inds=n_inds,
        k=k,
        n_output_inds=n_output_inds,
        n_cc=n_cc,
        randomize_names=randomize_names,
        seed=random_seed)

    # Get all inds
    all_inds = frozenset(mit.flatten(ts_inds))

    # Get random dimensions
    dims = dict(zip(all_inds, rng.choices(range(1, 3), k=len(all_inds))))

    # Get random arrays
    arrays = list(
        np_rng.normal(size=tuple(map(dims.get, xs))) for xs in ts_inds)

    # Get subscripts
    subscripts = get_einsum_subscripts(ts_inds, list(output_inds))

    # Get path
    path, _ = oe.contract_path(subscripts, *arrays)

    # Skip if too large
    if ContractionTree(path, ts_inds, dims,
                       output_inds=output_inds).max_width() > 25:
        pytest.skip("Contraction is too large.")

    # Contract using einsum
    merged_array_1 = oe.contract(subscripts, *arrays, optimize=path)

    # Contract using tnco
    [merged_inds], merged_output_inds, [merged_array_2
                                       ] = contract(path, ts_inds, output_inds,
                                                    arrays)
    assert frozenset(merged_inds) == merged_output_inds
    merged_array_2 = merged_array_2.transpose(
        list(map(merged_inds.index, list(output_inds))))

    # Check
    np.testing.assert_allclose(merged_array_1, merged_array_2, atol=1e-5)


@pytest.mark.timeout(30)
@pytest.mark.usefixtures("timeout")
@repeat(20)
def test_Fuse(random_seed, **kwargs):
    # Fuse tensors
    def fuse_(arrays, path, fused_inds):
        for (px, py), iz in zip(path, fused_inds):
            # Should be ordered
            assert px >= 0 and py > 0 and px < py

            # Get tensors and inds
            ty = arrays.pop(py)
            tx = arrays.pop(px)

            # Contract
            arrays.append(
                Tensor(
                    np.einsum(
                        tensor_utils.get_einsum_subscripts(
                            tx.inds, ty.inds, iz), tx.data, ty.data), iz))

        # Get final fused tensor
        return arrays

    # Initialize RNG
    rng = Random(random_seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(10, 15))
    n_inds = kwargs.get('n_inds', rng.randint(15, 30))
    k = kwargs.get('k', rng.randint(2, 5))
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, 5))
    n_cc = kwargs.get('n_cc', rng.randint(1, 4))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))

    # Initialize numpy rng
    np_rng = np.random.default_rng(random_seed)

    # Get tensors
    try:
        ts_inds, output_inds = generate_random_tensors(
            n_tensors,
            n_inds,
            k,
            n_output_inds=n_output_inds,
            n_cc=n_cc,
            randomize_names=randomize_names,
            seed=random_seed)
    except ValueError as e:
        if str(e) == "Too few indices.":
            pytest.skip(str(e))
        raise e

    # Get random dimensions
    dims = dict(
        map(lambda x:
            (x, rng.randrange(1, 5)), mit.unique_everseen(
                mit.flatten(ts_inds)))) if rng.randrange(2) else rng.randrange(
                    1, 5)

    # How to get the weight
    def get_width(xs):
        return log2(dims) * len(xs) if isinstance(dims, int) else sum(
            map(log2, map(dims.get, xs)))

    if max(map(get_width, ts_inds)) > 16:
        pytest.skip("Too large tensors")

    if max(map(len, ts_inds)) > 32:
        pytest.skip("Too many indices")

    # Set max weight
    max_width = kwargs.get(
        'max_width',
        2 * np.median(np.fromiter(map(get_width, ts_inds), dtype=float)))

    # Generate random arrays
    arrays = list(
        its.starmap(
            lambda xs, shape: Tensor(np_rng.normal(size=shape), xs),
            map(
                lambda xs:
                (xs, (dims,) * len(xs)
                 if isinstance(dims, int) else tuple(map(dims.get, xs))),
                ts_inds)))
    arrays = list(map(lambda a: a / np.sqrt(np.linalg.norm(a.data)), arrays))

    # Get exact result
    ex_tensor = TensorNetwork(arrays).contract(all, output_inds=output_inds)

    # if k > 2, callling without output_inds should raise
    if (k > 2):
        try:
            fuse(ts_inds, dims, max_width)
        except ValueError as e:
            assert str(
                e
            ) == "'output_inds' must be provided if " \
                 "'ts_inds' has hyper-indices."

    # Fuse with zero width
    path, fused_inds = fuse(ts_inds,
                            dims + 1 if isinstance(dims, int) else dict(
                                its.starmap(lambda x, d:
                                            (x, d + 1), dims.items())),
                            0,
                            output_inds=output_inds,
                            seed=random_seed,
                            return_fused_inds=True)
    assert len(path) == len(fused_inds) == 0

    # Fuse with given width
    path, fused_inds = fuse(ts_inds,
                            dims,
                            max_width,
                            output_inds=output_inds,
                            seed=random_seed,
                            return_fused_inds=True)
    # If k = 2, it should return the same results, even if output_inds are not
    # provided
    if k == 2:
        assert fuse(ts_inds,
                    dims,
                    max_width,
                    seed=random_seed,
                    return_fused_inds=True) == (path, fused_inds)

    # Calling fuse twice should give the same result
    assert fuse(ts_inds,
                dims,
                max_width,
                output_inds=output_inds,
                seed=random_seed,
                return_fused_inds=True) == (path, fused_inds)

    if max(map(get_width, fused_inds)) <= 16:
        # Get final fused tensor
        fsd_tensors = fuse_(arrays[:], path, fused_inds)

        # Check connected components
        assert n_output_inds == 0 or len(
            get_connected_components(map(lambda x: x.inds,
                                         fsd_tensors))) == n_cc

        # Contract them
        ctr_tensor = TensorNetwork(fsd_tensors).contract(
            all, output_inds=output_inds)

        # Check output inds
        assert n_output_inds == 0 or frozenset(
            ctr_tensor.inds) == frozenset(output_inds)

        # Check contraction
        np.testing.assert_allclose(float(ctr_tensor) -
                                   float(ex_tensor) if n_output_inds == 0 else
                                   (ctr_tensor - ex_tensor).data,
                                   0,
                                   atol=1e-5)

    # Fuse with infinite width
    path, fused_inds = fuse(ts_inds,
                            dims,
                            float('inf'),
                            output_inds=output_inds,
                            seed=random_seed,
                            return_fused_inds=True)

    if max(map(get_width, fused_inds)) <= 16:
        # Get final fused tensor
        fsd_tensors = fuse_(arrays[:], path, fused_inds)

        # Check connected components
        assert n_output_inds == 0 or len(fsd_tensors) == n_cc and len(
            get_connected_components(map(lambda x: x.inds,
                                         fsd_tensors))) == n_cc

        # Contract them
        ctr_tensor = TensorNetwork(fsd_tensors).contract(
            all, output_inds=output_inds)

        # Check output inds
        assert n_output_inds == 0 or frozenset(
            ctr_tensor.inds) == frozenset(output_inds)

        # Check contraction
        np.testing.assert_allclose(float(ctr_tensor) -
                                   float(ex_tensor) if n_output_inds == 0 else
                                   (ctr_tensor - ex_tensor).data,
                                   0,
                                   atol=1e-5)


@repeat(20)
def test_TensorSVD(random_seed, **kwargs):
    # Get generator
    rng = Random(random_seed)
    np_rng = np.random.default_rng(random_seed)

    # Set params
    n_inds = kwargs.get('n_inds', rng.randint(1, 6))
    n_left_inds = kwargs.get('n_left_inds', rng.randint(0, n_inds))

    # Get random indices
    *inds, svd_index_name = generate_random_inds(n_inds + 1, seed=random_seed)
    inds = tuple(inds)
    svd_index_name = None if rng.randint(0, 1) else svd_index_name
    left_inds = tuple(rng.sample(inds, k=n_left_inds))
    right_inds = tuple(filter(lambda x: x not in left_inds, inds))

    # Get random dimensions
    dims = dict(zip(inds, rng.choices(range(1, 6), k=len(inds))))

    # Generate random array
    array = np_rng.normal(size=tuple(map(dims.get, inds)))

    # This should fail
    try:
        tensor_utils.svd(array,
                         inds,
                         left_inds,
                         svd_index_name=rng.choice(inds))
    except ValueError as e:
        assert str(e) == "'svd_index_name' must be different from 'inds'."

    # Decompose array
    d_arrays = tensor_utils.svd(array,
                                inds,
                                left_inds,
                                svd_index_name=svd_index_name)

    # Check if correct
    np.testing.assert_allclose(TensorNetwork(
        map(lambda x: Tensor(*x),
            d_arrays)).contract(output_inds=inds).transpose(*inds).data,
                               array,
                               atol=1e-5)
    assert len(d_arrays) == 1 or (d_arrays[2][1][1:] == right_inds and
                                  d_arrays[0][1][:-1] == left_inds)

    # Check the case in which left_inds is empty
    d_arrays = tensor_utils.svd(array, inds, (), svd_index_name=svd_index_name)
    np.testing.assert_allclose(d_arrays[0][0], array, atol=1e-5)
    assert len(d_arrays) == 1 and d_arrays[0][1] == inds

    # Check the case in which left_inds is a permutation of inds
    new_inds = tuple(rng.sample(inds, k=len(inds)))
    d_arrays = tensor_utils.svd(array,
                                inds,
                                new_inds,
                                svd_index_name=svd_index_name)
    np.testing.assert_allclose(d_arrays[0][0].transpose(
        tuple(map(new_inds.index, inds))),
                               array,
                               atol=1e-5)
    assert len(d_arrays) == 1 and d_arrays[0][1] == new_inds


@repeat(20)
def test_GetLargestIntermediate(random_seed: int, **kwargs):
    # Initialize RNG
    rng = Random(random_seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(100, 300))
    n_inds = kwargs.get('n_inds', rng.randint(300, 500))
    k = kwargs.get('k', rng.randint(2, 6))
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, 150))
    n_cc = 1
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))

    # Get tensors and the dimension of the inds
    try:
        ts_inds, output_inds = generate_random_tensors(
            n_tensors,
            n_inds,
            k,
            n_output_inds=n_output_inds,
            n_cc=n_cc,
            randomize_names=randomize_names,
            seed=random_seed)
    except ValueError as e:
        if str(e) == "Too few indices.":
            pytest.skip(str(e))
        raise e

    # Get random dimensions
    dims = dict(
        map(lambda x:
            (x, rng.randint(1, 2)), mit.unique_everseen(
                mit.flatten(ts_inds)))) if rng.randrange(2) else rng.randint(
                    1, 2)

    # How to get the width
    def width(xs):
        return len(xs) * log2(dims) if isinstance(dims, int) else sum(
            map(log2, map(dims.get, xs)))

    # Get contraction
    paths = get_random_contraction_path(ts_inds,
                                        seed=random_seed,
                                        merge_paths=False)
    assert len(paths) == 1

    # Get contraction tree
    ctree = ContractionTree(paths[0], ts_inds, dims, output_inds=output_inds)

    # Get widths
    max_width = ctree.max_width()
    ex_max_width = max(map(width, ctree.inds))

    # Check
    assert abs(max_width - ex_max_width) < 1e-5


@repeat(40)
def test_DecomposeHyperIndsTN(random_seed: int, **kwargs):
    # Initialize RNG
    rng = Random(random_seed)
    np_rng = np.random.default_rng(seed=random_seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(10, 20))
    n_inds = kwargs.get('n_inds', rng.randint(20, 40))
    k = kwargs.get('k', rng.randint(2, 3))
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, 10))
    n_cc = 1
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))

    # Get tensors and the dimension of the inds
    try:
        ts_inds, output_inds = generate_random_tensors(
            n_tensors,
            n_inds,
            k,
            n_output_inds=n_output_inds,
            n_cc=n_cc,
            randomize_names=randomize_names,
            seed=random_seed)
    except ValueError as e:
        if str(e) == "Too few indices.":
            pytest.skip(str(e))
        raise e

    # Get random dimensions
    dims = dict(
        map(lambda x:
            (x, rng.randint(2, 3)), mit.unique_everseen(
                mit.flatten(ts_inds)))) if rng.randrange(2) else rng.randint(
                    2, 3)

    # Get contraction
    paths = get_random_contraction_path(ts_inds,
                                        seed=random_seed,
                                        merge_paths=False)
    assert len(paths) == 1

    # Skip if too large
    if ContractionTree(paths[0], ts_inds, dims,
                       output_inds=output_inds).max_width() > 28:
        pytest.skip("Contraction is too large.")

    def gen_array(shape):

        # Split inds witht thte same dimension
        dims_map = tuple(
            mit.map_reduce(enumerate(shape), op.itemgetter(1),
                           op.itemgetter(0)).items())

        # Randomly choose one dimension and randomly select which will be
        # hyper-inds
        hyper_dim, hyper_pos = rng.choice(dims_map)
        hyper_pos = rng.sample(hyper_pos, k=rng.randint(1, len(hyper_pos)))

        # Get permutation so that the first inds are hyper-inds
        perm = hyper_pos + list(
            filter(lambda x: x not in hyper_pos, range(len(shape))))
        perm_shape = tuple(map(lambda x: shape[x], perm))

        # Generate array
        array = np.zeros(perm_shape)

        # Set only diagonal tems
        array[(range(hyper_dim),) * len(hyper_pos)] = np_rng.normal(
            size=perm_shape[len(hyper_pos) - 1:])

        # Transpose
        array = array.transpose(tuple(map(perm.index, range(len(shape)))))
        assert array.shape == shape

        # Return array
        return array

    # Generate random arrays
    tn = TensorNetwork(
        map(
            lambda xs: Tensor(
                gen_array((dims,) * len(xs) if isinstance(dims, int) else tuple(
                    map(dims.get, xs))), xs), ts_inds))

    # Get contraction without decomposing, and decompose only at the end
    array_1 = tn.contract(output_inds=output_inds)
    if len(output_inds):
        array_1 = (lambda x: (Tensor(*x[0]), x[1]))(tensor_decompose_hyper_inds(
            array_1.data, array_1.inds))[0]

    # Decompose tn
    new_arrays, new_ts_inds, hyper_inds_map_2 = tn_decompose_hyper_inds(
        *mit.transpose(map(lambda t: (t.data, t.inds), tn)))

    # Get new output inds
    new_output_inds = frozenset(map(hyper_inds_map_2.get, output_inds))

    # Get contraction with decomposition
    array_2 = TensorNetwork(
        map(lambda x: Tensor(*x),
            zip(new_arrays, new_ts_inds))).contract(output_inds=new_output_inds)

    if len(output_inds):
        # Rename to match axes
        array_1 = array_1.reindex(
            dict(
                map(lambda x: (x, hyper_inds_map_2[x]),
                    frozenset(array_1.inds).difference(array_2.inds))))

        # Transpose
        array_2 = array_2.transpose_like(array_1)

    # Check
    array_1 = array_1 if isinstance(array_1, float) else array_1.data
    array_2 = array_1 if isinstance(array_2, float) else array_2.data
    np.testing.assert_allclose(array_1, array_2, atol=1e-5)


@repeat(40)
def test_merge_contraction_paths(random_seed, **kwargs):
    # Initialize random number generator
    rng = Random(random_seed)

    # Initialize random number generator (numpy)
    np_rng = np.random.default_rng(random_seed)

    # Set parameters
    n_tensors = kwargs.get('n_tensors', rng.randint(5, 10))
    k = kwargs.get('k', rng.randint(2, 4))
    n_inds = kwargs.get('n_inds', rng.randint(8, 15))
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, n_inds // 5))
    n_cc = kwargs.get('n_cc', rng.randint(1, 3))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))

    # Check minimum number of indices
    if (n_inds - n_output_inds) < n_tensors + 1 - k:
        pytest.skip("Too few indices")

    # Get tensors
    ts_inds, output_inds = generate_random_tensors(
        n_tensors=n_tensors,
        n_inds=n_inds,
        k=k,
        n_cc=n_cc,
        n_output_inds=n_output_inds,
        randomize_names=randomize_names,
        seed=random_seed)

    # Get random dimensions
    dims = dict(
        map(lambda x: (x, rng.randint(2, 3)),
            mit.unique_everseen(mit.flatten(ts_inds))))

    # Get random arrays
    arrays = list(
        map(lambda xs: np_rng.normal(0, 1, size=tuple(map(dims.get, xs))),
            ts_inds))

    # Get paths
    paths = get_random_contraction_path(ts_inds,
                                        seed=random_seed,
                                        merge_paths=False)

    # Get tensor network
    tn = TensorNetwork(map(Tensor, arrays, ts_inds))

    # Use optimized path
    array_1 = tn.contract(output_inds=output_inds)
    array_2 = tn.contract(output_inds=output_inds,
                          optimize=merge_contraction_paths(len(ts_inds), paths))

    # Check
    if isinstance(array_1, float):
        np.testing.assert_allclose(array_1, array_2, atol=1e-5)
    else:
        np.testing.assert_allclose(array_1.transpose_like(array_2).data,
                                   array_2.data,
                                   atol=1e-5)


@repeat(40)
def test_split_contraction_path(random_seed, **kwargs):
    # Initialize random number generator
    rng = Random(random_seed)

    # Set parameters
    n_tensors = kwargs.get('n_tensors', rng.randint(150, 300))
    k = kwargs.get('k', rng.randint(2, 4))
    n_inds = kwargs.get('n_inds', rng.randint(240, 450))
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, n_inds // 5))
    n_cc = kwargs.get('n_cc', rng.randint(1, 5))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))

    # Check minimum number of indices
    if (n_inds - n_output_inds) < n_tensors + 1 - k:
        pytest.skip("Too few indices")

    # Get tensors
    ts_inds, _ = generate_random_tensors(n_tensors=n_tensors,
                                         n_inds=n_inds,
                                         k=k,
                                         n_cc=n_cc,
                                         n_output_inds=n_output_inds,
                                         randomize_names=randomize_names,
                                         seed=random_seed)

    # Get path
    path = get_random_contraction_path(ts_inds,
                                       seed=random_seed,
                                       autocomplete=False)

    # Split paths
    paths, cc = split_contraction_path(n_tensors * n_cc,
                                       path,
                                       return_connected_components=True)
    assert len(paths) == len(cc)
    assert all(isinstance(cc, frozenset) for cc in cc)
    cc = sorted(map(tuple, map(sorted, cc)))

    # The merged path should be identical to the original one
    assert merge_contraction_paths(n_tensors * n_cc, paths,
                                   autocomplete=False) == path

    # Get connected components from inds
    cc_from_inds = sorted(
        map(tuple, map(sorted, get_connected_components(ts_inds))))
    assert cc_from_inds == cc

    # Get normalized paths
    normal_paths, cc_ = split_contraction_path(n_tensors * n_cc,
                                               path,
                                               return_connected_components=True,
                                               normalize_paths=True)
    assert len(normal_paths) == len(cc_)
    cc_ = sorted(map(tuple, map(sorted, cc_)))
    assert cc == cc_

    # Check contractions
    for normal_tensors, path, normal_path in zip(map(list, cc), paths,
                                                 normal_paths):
        # Normal and non-normal paths should have the same number of
        # contractions
        assert len(path) == len(normal_path)

        # Add extra tag
        normal_tensors = list(zip(normal_tensors, its.repeat('__CHECK__')))

        # The non-normal path should have all the initial tensors
        tensors = list(zip(range(n_tensors * n_cc), its.repeat('__CHECK__')))
        n_intermediate_tensors = n_tensors

        # For each contraction ...
        for (x, y), (nx, ny) in zip(map(sorted, path), map(sorted,
                                                           normal_path)):
            # Normal and non-normal path should point to the same tensor
            assert tensors.pop(y) == normal_tensors.pop(ny)
            assert tensors.pop(x) == normal_tensors.pop(nx)

            # Append the intermediate tensors
            tensors.append((n_intermediate_tensors, '__CHECK__'))
            normal_tensors.append((n_intermediate_tensors, '__CHECK__'))

            # Update the number of intermediate tensors
            n_intermediate_tensors += 1


@repeat(20)
def test_Contract(random_seed, **kwargs):
    # Initialize RNG
    rng = Random(random_seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(10, 15))
    n_inds = kwargs.get('n_inds', rng.randint(15, 30))
    k = kwargs.get('k', rng.randint(2, 5))
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, 5))
    n_cc = kwargs.get('n_cc', rng.randint(1, 4))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))

    # Initialize numpy rng
    np_rng = np.random.default_rng(random_seed)

    # Get tensors
    try:
        ts_inds, output_inds = generate_random_tensors(
            n_tensors,
            n_inds,
            k,
            n_output_inds=n_output_inds,
            n_cc=n_cc,
            randomize_names=randomize_names,
            seed=random_seed)
    except ValueError as e:
        if str(e) == "Too few indices.":
            pytest.skip(str(e))
        raise e

    # Get random dimensions
    dims = dict(
        map(lambda x:
            (x, rng.randrange(1, 5)), mit.unique_everseen(
                mit.flatten(ts_inds)))) if rng.randrange(2) else rng.randrange(
                    1, 5)

    # How to get the weight
    def get_width(xs):
        return log2(dims) * len(xs) if isinstance(dims, int) else sum(
            map(log2, map(dims.get, xs)))

    if max(map(get_width, ts_inds)) > 16:
        pytest.skip("Too large tensors")

    if max(map(len, ts_inds)) > 32:
        pytest.skip("Too many indices")

    # Set max weight
    max_width = kwargs.get(
        'max_width',
        2 * np.median(np.fromiter(map(get_width, ts_inds), dtype=float)))

    # Generate random arrays
    arrays = list(
        its.starmap(
            lambda xs, shape: Tensor(np_rng.normal(size=shape), xs),
            map(
                lambda xs:
                (xs, (dims,) * len(xs)
                 if isinstance(dims, int) else tuple(map(dims.get, xs))),
                ts_inds)))
    arrays = list(map(lambda a: a / np.sqrt(np.linalg.norm(a.data)), arrays))

    # Fuse with given width
    path = fuse(ts_inds,
                dims,
                max_width,
                output_inds=output_inds,
                seed=random_seed)

    # Get tensor network from fused arrays
    fused_ts_inds, fused_output_inds, fused_arrays = contract(
        path,
        ts_inds,
        output_inds=output_inds,
        arrays=map(lambda t: t.data, arrays))
    assert fused_output_inds.issubset(output_inds)
    fused_tn = TensorNetwork(
        map(Tensor, fused_arrays,
            fused_ts_inds)).contract(output_inds=fused_output_inds)

    # Get exact tn
    exact_tn = TensorNetwork(arrays).contract(output_inds=output_inds)

    # Check
    if isinstance(exact_tn, Tensor):
        np.testing.assert_allclose(exact_tn.transpose_like(fused_tn).data,
                                   fused_tn.data,
                                   atol=1e-5)
    else:
        np.testing.assert_allclose(exact_tn, fused_tn, atol=1e-5)


@repeat(200)
def test_OrderedFrozenSet(random_seed):
    # Get rng
    rng = Random(random_seed)

    # Generate random inds
    n = 500
    x = generate_random_inds(rng.randrange(1, n), seed=rng.randrange(2**32))
    n_shared = rng.randrange(len(x))
    y = tuple(
        its.chain(
            rng.sample(x, k=n_shared),
            generate_random_inds(rng.randrange(1, n - n_shared),
                                 seed=rng.randrange(2**32))))
    sub_x = rng.sample(x, rng.randrange(len(x)))

    # Get ordered set
    ordered_set_x = OrderedFrozenSet(rng.sample(x, k=len(x)))
    ordered_set_y = OrderedFrozenSet(rng.sample(y, k=len(y)))

    # Get frozensets
    set_x = frozenset(rng.sample(x, k=len(x)))
    set_y = frozenset(rng.sample(y, k=len(y)))

    # Checks
    assert ordered_set_x == set_x
    assert ordered_set_y == set_y
    assert not ordered_set_x != set_x
    assert not ordered_set_y != set_y
    assert (ordered_set_x < ordered_set_y) == (set_x < set_y)
    assert (ordered_set_x > ordered_set_y) == (set_x > set_y)
    assert (ordered_set_x <= ordered_set_y) == (set_x <= set_y)
    assert (ordered_set_x >= ordered_set_y) == (set_x >= set_y)
    assert (ordered_set_x & ordered_set_y) == (set_x & set_y)
    assert (ordered_set_x ^ ordered_set_y) == (set_x ^ set_y)
    assert (ordered_set_x | ordered_set_y) == (set_x | set_y)
    assert (ordered_set_x - ordered_set_y) == (set_x - set_y)
    assert ordered_set_x.issuperset(sub_x)
    assert not ordered_set_x.issubset(sub_x)
    assert OrderedFrozenSet(sub_x).issubset(ordered_set_x)
    assert not OrderedFrozenSet(sub_x).issuperset(ordered_set_x)
    assert ordered_set_x.isdisjoint(ordered_set_y) == set_x.isdisjoint(set_y)
    assert ordered_set_x.isdisjoint(sub_x) == set_x.isdisjoint(sub_x)
    assert ordered_set_x.isdisjoint(ordered_set_y) == set_x.isdisjoint(set_y)
    assert ordered_set_x.isdisjoint(sub_x) == set_x.isdisjoint(sub_x)

    # Check iterator order
    assert tuple(ordered_set_x) == ordered_set_x._order
    assert tuple(ordered_set_y) == ordered_set_y._order
    res = rng.sample(tuple(ordered_set_x), k=len(ordered_set_x))
    assert tuple(res) == tuple(OrderedFrozenSet(res))

    # Check order for -
    res = ordered_set_x - ordered_set_y
    assert tuple(res) == tuple(
        filter(lambda x: x not in ordered_set_y, ordered_set_x))
    res = ordered_set_y - ordered_set_x
    assert tuple(res) == tuple(
        filter(lambda y: y not in ordered_set_x, ordered_set_y))

    # Check order for &
    res = ordered_set_x & ordered_set_y
    assert tuple(res) == tuple(filter(lambda x: x in res, ordered_set_x))

    # Check order for |
    res = ordered_set_x | ordered_set_y
    assert sum(
        map(lambda x: x == -1,
            mit.difference(map(lambda z: z in ordered_set_x, res)))) == 1
    assert tuple(filter(lambda z: z in ordered_set_x,
                        res)) == tuple(ordered_set_x)
    assert tuple(
        filter(lambda z: z not in ordered_set_x and z in ordered_set_y,
               res)) == tuple(ordered_set_y - ordered_set_x)

    # Check order for ^
    res = ordered_set_x ^ ordered_set_y
    assert tuple(filter(lambda z: z in ordered_set_x,
                        res)) == tuple(ordered_set_x - ordered_set_y)
    assert tuple(filter(lambda z: z in ordered_set_y,
                        res)) == tuple(ordered_set_y - ordered_set_x)
