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
from copy import deepcopy
from math import exp, log2
from random import Random

import more_itertools as mit
import pytest
from tnco_core import Bitset as Bitset_
from tnco_core import ContractionTree as ContractionTree_
from tnco_core import Node
from tnco_core import Tree as Tree_
from tnco_core import float1024
from tnco_core.utils import all_close, all_logclose

from conftest import fraction_n_tests, global_seed  # Get global seed
from tnco.bitset import Bitset
from tnco.ctree import ContractionTree, traverse_tree
from tnco.optimize.finite_width.cost_model import \
    SimpleCostModel as FW_SimpleCostModel
from tnco.optimize.infinite_memory.cost_model import \
    SimpleCostModel as IM_SimpleCostModel
from tnco.optimize.prob import BaseProbability, Greedy, MetropolisHastings
from tnco.tests.utils import generate_random_inds, generate_random_tensors
from tnco.utils.tn import get_random_contraction_path

rng = Random(global_seed)


def sample_seeds(k, /):
    k = int(max(1, k * fraction_n_tests))
    return rng.sample(range(2**32), k=k)


@pytest.mark.parametrize('seed', sample_seeds(200))
def test_float1024(seed):
    # Check if two variables are close
    def isclose(x, y):
        return abs(x - y) < 1e-10

    # Initialize random number generator
    rng = Random(seed)

    # Initialize variables
    x = rng.normalvariate(0, 10)
    y = rng.normalvariate(0, 10)
    xf = float1024(x)
    yf = float1024(y)

    # Operations to test
    op_test = [
        op.add,
        op.sub,
        op.truediv,
        op.mul,
        op.le,
        op.lt,
        op.ge,
        op.gt,
        op.eq,
        op.ne,
        lambda x, y: abs(x),
        lambda x, y: +x,
        lambda x, y: -x,
        fts.partial(lambda x, y, r: abs(x), r=rng.normalvariate(0, 10)),
    ]

    # Check all operations
    assert all(
        map(
            fts.partial(lambda op, r: isclose(op(x, r), op(xf, r)),
                        r=rng.normalvariate(0, 10)), op_test))
    assert all(
        map(
            fts.partial(lambda op, r: isclose(op(r, x), op(r, xf)),
                        r=rng.normalvariate(0, 10)), op_test))
    assert all(map(lambda op: isclose(op(x, x), op(xf, xf)), op_test))
    assert all(map(lambda op: isclose(op(x, y), op(xf, yf)), op_test))
    assert all(
        map(lambda op: isclose(op(x, x), op(float(str(xf)), float(str(xf)))),
            op_test))
    assert all(
        map(lambda op: isclose(op(x, y), op(float(str(xf)), float(str(yf)))),
            op_test))

    # Check assignments
    x -= y
    xf -= yf
    assert isclose(x, xf)
    x *= y
    xf *= yf
    assert isclose(x, xf)
    x += y
    xf += yf
    assert isclose(x, xf)
    x /= y
    xf /= yf
    assert isclose(x, xf)
    x -= y
    xf -= y
    assert isclose(x, xf)
    x *= y
    xf *= y
    assert isclose(x, xf)
    x += y
    xf += y
    assert isclose(x, xf)
    x /= y
    xf /= y
    assert isclose(x, xf)


@pytest.mark.parametrize('seed', sample_seeds(200))
def test_Bitset(seed: int, **kwargs):
    # Get random number generator
    rng = Random(seed)

    # Initialize variables
    n_pos = kwargs.get('n_pos', rng.randint(100, 1_000))
    n_vars = kwargs.get('n_vars', rng.randint(1_000, 2_000))

    # Get positions
    pos = rng.sample(range(n_vars), k=n_pos)

    # Get bitset
    bitset = Bitset(pos, n_vars)

    # Check sizes
    assert len(bitset) == n_vars
    assert bitset.count() == n_pos

    # Trigger an error if argument not in pos
    def check_(x):
        assert x in pos

    # Check visit
    bitset.visit(check_)

    # Check equality
    assert bitset != Bitset([], 0) and bitset == pickle.loads(
        pickle.dumps(bitset))

    # Generate random sets
    sets = [
        frozenset(rng.sample(range(n_vars), k=n_vars // 2)) for _ in range(1000)
    ]

    # Check bitwise operations
    for x_, y_ in zip(rng.sample(range(1000), 100),
                      rng.sample(range(1000), 100)):
        assert Bitset(
            sets[x_] & sets[y_],
            n_vars) == Bitset(sets[x_], n_vars) & Bitset(sets[y_], n_vars)
        assert Bitset(
            sets[x_] | sets[y_],
            n_vars) == Bitset(sets[x_], n_vars) | Bitset(sets[y_], n_vars)
        assert Bitset(
            sets[x_] ^ sets[y_],
            n_vars) == Bitset(sets[x_], n_vars) ^ Bitset(sets[y_], n_vars)
        assert Bitset(
            sets[x_] - sets[y_],
            n_vars) == Bitset(sets[x_], n_vars) - Bitset(sets[y_], n_vars)
        assert Bitset(frozenset(range(n_vars)) - sets[x_],
                      n_vars) == ~Bitset(sets[x_], n_vars)

    # Initialize random positions
    bs = rng.sample(range(n_vars), k=n_pos)
    str_bs = ''.join(map(lambda x: '1' if x in bs else '0', range(n_vars)))

    # Check constructors
    assert Bitset(bs, n_vars) == Bitset(str_bs)
    assert str(Bitset(bs, n_vars)) == str(Bitset(str_bs)) == str_bs

    # Check visit
    bs_ = []
    Bitset(bs, n_vars).visit(lambda x: bs_.append(x))
    assert bs_ == sorted(bs)
    assert bs_ == Bitset(bs, n_vars).positions()

    # Check __getitem__
    bs_ = Bitset(bs, n_vars)
    assert all(map(lambda x: bs_[x] == int(str_bs[x]), range(n_vars)))
    assert all(map(lambda x, y: x == int(y), bs_, str_bs))


@pytest.mark.parametrize('seed', sample_seeds(200))
def test_Node(seed: int):
    # Check empty node
    empty_node = Node()
    assert empty_node.is_leaf()
    assert empty_node.is_root()
    assert empty_node.parent is None
    assert empty_node.children == (None, None)

    # Initialize RNG
    rng = Random(seed)

    def check_node_1(children, parent):
        if (children[0] >= 0) ^ (children[1] >= 0) or children[0] == children[
                1] or parent == children[0] or parent == children[1]:
            try:
                Node(children, parent)
            except ValueError:
                return True
            except Exception:
                return False

        node = Node(children, parent)
        if node.children != (None if children[0] < 0 else children[0],
                             None if children[1] < 0 else children[1]):
            return False

        if node.parent != (None if parent < 0 else parent):
            return False

        return node.is_valid()

    def check_node_2(children):
        if (children[0] >= 0) ^ (children[1]
                                 >= 0) or children[0] == children[1]:
            try:
                Node(children)
            except ValueError:
                return True
            except Exception:
                return False

        node = Node(children)
        if node.children != (None if children[0] < 0 else children[0],
                             None if children[1] < 0 else children[1]):
            return False

        return node.is_valid()

    def check_node_3(parent):
        node = Node(parent)

        if node.parent != (None if parent < 0 else parent):
            return False

        return node.is_valid()

    # Check
    assert all(
        check_node_1((rng.randint(-1000, 1000),
                      rng.randint(-1000, 1000)), rng.randint(-1000, 1000))
        for _ in range(10_000))
    assert all(
        check_node_2((rng.randint(-1000, 1000), rng.randint(-1000, 1000)))
        for _ in range(10_000))
    assert all(check_node_3(rng.randint(-1000, 1000)) for _ in range(10_000))


@pytest.mark.parametrize('seed', sample_seeds(200))
def test_ContractionTree(seed: int, **kwargs):
    # Check empty ContractionTree
    empty_ctree = ContractionTree_()
    assert empty_ctree.is_valid()
    assert empty_ctree.dims == 1
    assert empty_ctree.inds == [Bitset_()]
    assert empty_ctree.n_inds == 0
    assert empty_ctree.n_leaves == 1
    assert empty_ctree.nodes == [Node()]

    # Initialize RNG
    rng = Random(seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(100, 300))
    n_inds = kwargs.get('n_inds', rng.randint(300, 1000))
    k = kwargs.get('k', rng.randint(2, 10))
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, 100))
    n_cc = kwargs.get('n_cc', rng.randint(1, 5))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))
    verbose = kwargs.get('verbose', False)
    seed = rng.randrange(2**32)

    # Get tensors
    try:
        tensors, output_inds = generate_random_tensors(
            n_tensors=n_tensors,
            n_inds=n_inds,
            k=k,
            n_output_inds=n_output_inds,
            n_cc=n_cc,
            randomize_names=randomize_names,
            seed=seed,
            verbose=verbose)
    except ValueError as e:
        if str(e) == "Too few indices.":
            pytest.skip(str(e))
        raise e

    # Get random dimensions
    dims = dict(
        map(lambda x:
            (x, rng.randrange(1, 10)), mit.unique_everseen(
                mit.flatten(tensors)))) if rng.randrange(2) else rng.randrange(
                    1, 8)

    # Get contraction
    paths = get_random_contraction_path(tensors,
                                        output_inds,
                                        seed=seed,
                                        verbose=verbose,
                                        merge_paths=False)

    # Get random contraction trees
    ctrees = [
        Tree_(
            ContractionTree(path_,
                            tensors,
                            dims,
                            output_inds=output_inds,
                            check_shared_inds=True)) for path_ in paths
    ]

    # Check traverse_tree
    def test_TraverseTree(ctree):
        all_pos = set(range(len(ctree.nodes)))

        def check(pos):
            node = ctree.nodes[pos]
            assert pos in all_pos
            assert node.children[0] is None or node.children[0] not in all_pos
            assert node.children[1] is None or node.children[1] not in all_pos
            assert node.parent is None or node.parent in all_pos
            all_pos.remove(pos)

        traverse_tree(ctree, check)

    # Check positions
    mit.consume(map(test_TraverseTree, ctrees))

    # Copy trees
    ctrees_ = deepcopy(ctrees)

    # Check copy
    assert ctrees_ == ctrees and all(
        map(lambda ctree: ctree.is_valid(), ctrees_))

    # Randomly apply swap
    mit.consume(
        map(
            lambda ctree: mit.consume(
                map(ctree.swap_with_nn, rng.choices(range(len(ctree)), k=1000))
            ), ctrees))

    # Check if still valid
    assert all(map(lambda ctree: ctree.is_valid(), ctrees))

    # Check if deepcopy worked (this might fail for small tn)
    assert all(map(lambda x, y: x != y, ctrees_, ctrees))


@pytest.mark.parametrize('seed', sample_seeds(400))
def test_SimpleCostModel(seed, **kwargs):

    def rel_error(x, y, *, atol=1e-5):
        if x == 0 or y == 0:
            return x == y == 0
        return abs(x - y) / min(x, y) < atol

    # Compute cost
    def get_cost(x, y, z, dims, sparse_inds=(), n_projs=1):

        # Convert to the right type
        x = frozenset(x)
        y = frozenset(y)
        z = frozenset(z)
        all_inds = x | y
        sparse_inds = frozenset(sparse_inds)
        n_projs = int(n_projs)

        # Check consistency with z
        assert (x | y).issuperset(z)

        # Get cost depending if dims is a dictionary or an int
        def cost_(inds):
            if isinstance(dims, int):
                return dims**len(inds)
            else:
                return fts.reduce(op.mul, map(dims.get, inds), 1)

        # Return cost
        return min(cost_(all_inds & sparse_inds),
                   n_projs) * cost_(all_inds - sparse_inds)

    # Initialize rng
    rng = Random(seed)

    # Get params
    n_vars = kwargs.get('n_vars', rng.randrange(200, 500))
    n_pos = kwargs.get('n_pos', rng.randrange(50, 100))
    cost_type = kwargs.get('cost_type',
                           'float128' if rng.randrange(2) else 'float1024')

    # Get random inds
    inds = generate_random_inds(n_vars)

    # Get random dimensions
    dims = rng.randrange(1, 10) if rng.randrange(2) else dict(
        zip(inds, rng.choices(range(1, 10), k=n_vars)))

    # Check types
    assert all(
        IM_SimpleCostModel(
            cost_type=f'{ct_}').__get_core__('').cost_type == f'{ct_}'
        for ct_ in ('float32', 'float64', 'float128', 'float1024'))
    assert all(
        IM_SimpleCostModel(sparse_inds=(), n_projs=None, cost_type=f'{ct_}').
        __get_core__('').cost_type == f'{ct_}'
        for ct_ in ('float32', 'float64', 'float128', 'float1024'))

    # How to get the model
    def get_model(*args, **kwargs):
        return IM_SimpleCostModel(*args, cost_type=cost_type, **kwargs)

    # How to get random indices
    def random_xyz_inds():
        x = rng.sample(inds, k=n_pos)
        y = rng.sample(x, k=n_pos // 2)
        y += rng.sample(list(frozenset(inds).difference(x)), k=n_pos // 2)
        z = rng.sample(list(frozenset(x) & frozenset(y)),
                       k=n_pos // 4) + list(frozenset(x) ^ frozenset(y))
        return x, rng.sample(y, k=len(y)), rng.sample(z, k=len(z))

    # Check without sparse inds
    for xs_ in (random_xyz_inds() for _ in range(100)):
        assert rel_error(get_model().contraction_cost(*xs_, dims),
                         get_cost(*xs_, dims))
        assert rel_error(
            pickle.loads(pickle.dumps(
                (get_model()))).contraction_cost(*xs_, dims),
            get_cost(*xs_, dims))

    # These should fail
    try:
        IM_SimpleCostModel(sparse_inds=(1,))
    except ValueError as e:
        assert str(
            e) == "'n_projs' must be specified if 'sparse_inds' is provided."
    else:
        assert False
    try:
        IM_SimpleCostModel(n_projs=123)
    except ValueError as e:
        assert str(
            e
        ) == "'n_projs' cannot be specified if 'sparse_inds' is not provided."
    else:
        assert False

    # Check with sparse inds
    for xs_ in (random_xyz_inds() for _ in range(100)):
        sparse_ = rng.sample(tuple(fts.reduce(op.or_, map(frozenset, xs_))),
                             k=rng.randrange(1, 10))
        n_projs_ = rng.randrange(1, 100)
        assert rel_error(
            get_model(sparse_inds=sparse_,
                      n_projs=n_projs_).contraction_cost(*xs_, dims),
            get_cost(*xs_, dims, sparse_inds=sparse_, n_projs=n_projs_))
        assert rel_error(
            pickle.loads(
                pickle.dumps(get_model(sparse_inds=sparse_,
                                       n_projs=n_projs_))).contraction_cost(
                                           *xs_, dims),
            get_cost(*xs_, dims, sparse_inds=sparse_, n_projs=n_projs_))
        max_n_projs_ = dims**len(sparse_) if isinstance(
            dims, int) else fts.reduce(op.mul, map(dims.get, sparse_), 1)
        assert rel_error(
            get_model(sparse_inds=sparse_,
                      n_projs=max_n_projs_).contraction_cost(*xs_, dims),
            IM_SimpleCostModel(cost_type=cost_type).contraction_cost(
                *xs_, dims))


@pytest.mark.parametrize('seed', sample_seeds(400))
def test_SimpleCostModelFiniteWidth(seed, **kwargs):

    def rel_error(x, y, *, atol=1e-5):
        if x == 0 or y == 0:
            return x == y == 0
        return abs(x - y) / min(x, y) < atol

    def abs_error(x, y, *, atol=1e-5):
        return abs(x - y) < atol

    # Compute cost
    def get_cost(x, y, z, dims, slices=(), sparse_inds=(), n_projs=1):

        # Convert to the right type
        x = frozenset(x)
        y = frozenset(y)
        z = frozenset(z)
        slices = frozenset(slices)
        all_inds = x | y | slices
        sparse_inds = frozenset(sparse_inds)
        n_projs = int(n_projs)

        # Check consistency with z
        assert (x | y).issuperset(z)

        # Get cost depending if dims is a dictionary or an int
        def cost_(inds):
            if isinstance(dims, int):
                return dims**len(inds)
            else:
                return fts.reduce(op.mul, map(dims.get, inds), 1)

        # Return cost
        return min(cost_(all_inds & sparse_inds),
                   n_projs) * cost_(all_inds - sparse_inds)

    def get_width(inds, dims, sparse_inds=(), n_projs=1, slices=None):

        def width_(inds):
            if isinstance(dims, int):
                return log2(dims) * len(inds)
            else:
                return fts.reduce(op.add, map(lambda x: log2(dims[x]), inds), 0)

        inds = frozenset(inds)
        sparse_inds = frozenset(sparse_inds)
        if slices is not None:
            inds -= frozenset(slices)
        return min(width_(inds & sparse_inds),
                   log2(n_projs)) + width_(inds - sparse_inds)

    # Initialize rng
    rng = Random(seed)

    # Get params
    n_vars = kwargs.get('n_vars', rng.randrange(200, 500))
    n_pos = kwargs.get('n_pos', rng.randrange(50, 100))
    cost_type = kwargs.get('cost_type',
                           'float128' if rng.randrange(2) else 'float1024')
    width_type = kwargs.get('width_type', 'float128')

    # Get random inds
    inds = generate_random_inds(n_vars)

    # Get random dimensions
    dims = rng.randrange(1, 10) if rng.randrange(2) else dict(
        zip(inds, rng.choices(range(1, 10), k=n_vars)))

    # Check types
    assert all(
        its.starmap(
            lambda ct, wt, m: m.cost_type == ct and m.width_type == wt,
            its.starmap(
                lambda ct, wt:
                (ct, wt, FW_SimpleCostModel(0, cost_type=ct, width_type=wt)),
                its.product(['float32', 'float64', 'float128', 'float1024'],
                            ['float32', 'float64', 'float128']))))
    assert all(
        its.starmap(
            lambda ct, wt, m: m.cost_type == ct and m.width_type == wt,
            its.starmap(
                lambda ct, wt: (ct, wt,
                                FW_SimpleCostModel(0,
                                                   sparse_inds=(),
                                                   n_projs=None,
                                                   cost_type=ct,
                                                   width_type=wt)),
                its.product(['float32', 'float64', 'float128', 'float1024'],
                            ['float32', 'float64', 'float128']))))

    # How to get the model
    def get_model(*args, **kwargs):
        return FW_SimpleCostModel(*args,
                                  cost_type=cost_type,
                                  width_type=width_type,
                                  **kwargs)

    # How to get random indices
    def random_xyzs_inds():
        x = rng.sample(inds, k=n_pos)
        y = rng.sample(x, k=n_pos // 2)
        y += rng.sample(list(frozenset(inds).difference(x)), k=n_pos // 2)
        z = rng.sample(list(frozenset(x) & frozenset(y)),
                       k=n_pos // 4) + list(frozenset(x) ^ frozenset(y))
        s = rng.sample(inds, k=n_pos)
        return x, rng.sample(y, k=len(y)), rng.sample(z, k=len(z)), s

    # Check assignment
    assert all(
        its.starmap(
            lambda w, sp, np, m: m.max_width == w and m.sparse_inds == sp and m.
            n_projs == np,
            its.starmap(
                lambda w, sp, np:
                (w, sp, np, get_model(w, sparse_inds=sp, n_projs=np)),
                zip(
                    mit.repeatfunc(rng.random, 100),
                    map(frozenset, mit.repeatfunc(generate_random_inds, 100,
                                                  10)),
                    mit.repeatfunc(rng.randrange, 100, 1, 1000)))))

    # Check without sparse inds
    for *xs_, slices_ in (random_xyzs_inds() for _ in range(100)):
        max_width_ = rng.randrange(0, 10)
        assert abs_error(
            get_model(float('inf')).width(xs_[0], dims),
            get_width(xs_[0], dims))
        assert rel_error(
            get_model(max_width_).contraction_cost(*xs_, dims, slices_),
            get_cost(*xs_, dims, slices=slices_))
        assert rel_error(
            pickle.loads(pickle.dumps(get_model(max_width_))).contraction_cost(
                *xs_, dims, slices_), get_cost(*xs_, dims, slices=slices_))

    # These should fail
    try:
        FW_SimpleCostModel(0, sparse_inds=(1,))
    except ValueError as e:
        assert str(
            e) == "'n_projs' must be specified if 'sparse_inds' is provided."
    else:
        assert False
    try:
        FW_SimpleCostModel(0, n_projs=123)
    except ValueError as e:
        assert str(
            e
        ) == "'n_projs' cannot be specified if 'sparse_inds' is not provided."
    else:
        assert False

    # Check with sparse inds
    for *xs_, slices_ in (random_xyzs_inds() for _ in range(100)):
        max_width_ = rng.randrange(0, 10)
        sparse_ = rng.sample(tuple(fts.reduce(op.or_, map(frozenset, xs_))),
                             k=rng.randrange(1, 10))
        n_projs_ = rng.randrange(1, 100)
        assert abs_error(
            get_model(float('inf'), sparse_inds=sparse_,
                      n_projs=n_projs_).width(
                          filter(lambda x: x not in slices_, xs_[0]), dims),
            get_width(xs_[0],
                      dims,
                      sparse_inds=sparse_,
                      n_projs=n_projs_,
                      slices=slices_))
        assert rel_error(
            get_model(max_width_, sparse_inds=sparse_,
                      n_projs=n_projs_).contraction_cost(*xs_, dims, slices_),
            get_cost(*xs_,
                     dims,
                     sparse_inds=sparse_,
                     n_projs=n_projs_,
                     slices=slices_))
        assert rel_error(
            pickle.loads(
                pickle.dumps(
                    get_model(max_width_, sparse_inds=sparse_,
                              n_projs=n_projs_))).contraction_cost(
                                  *xs_, dims, slices_),
            get_cost(*xs_,
                     dims,
                     sparse_inds=sparse_,
                     n_projs=n_projs_,
                     slices=slices_))
        max_n_projs_ = dims**len(sparse_) if isinstance(
            dims, int) else fts.reduce(op.mul, map(dims.get, sparse_), 1)
        assert rel_error(
            get_model(max_width_, sparse_inds=sparse_,
                      n_projs=max_n_projs_).contraction_cost(
                          *xs_, dims, slices_),
            get_model(max_width_).contraction_cost(*xs_, dims, slices_))

    # --- check delta_width ---

    # Generate random indices
    inds = set(generate_random_inds(n_vars))

    # Generate random dims
    dims = rng.randrange(1, 10) if rng.randrange(2) else dict(
        zip(inds, rng.choices(range(1, 10), k=n_vars)))

    # Initialize cost model
    cmodel = FW_SimpleCostModel(max_width=10 * rng.random(),
                                width_type='float128')

    # Check if correct
    for index in rng.sample(tuple(inds), k=len(inds)):
        # Get delta width
        delta_width_1 = cmodel.width(inds - {index}, dims) - cmodel.width(
            inds, dims)
        delta_width_2 = cmodel.delta_width(inds, dims, index)

        # Check if they're the same
        assert abs(delta_width_1 - delta_width_2) < 1e-6

        # Remove index from indices
        inds -= {index}

    # --- check delta_width with sparse inds ---

    # Generate random indices
    inds = set(generate_random_inds(n_vars))

    # Generate random dims
    dims = rng.randrange(1, 10) if rng.randrange(2) else dict(
        zip(inds, rng.choices(range(1, 10), k=n_vars)))

    # Get random sparse indices
    sparse_inds = rng.sample(tuple(inds), k=n_vars // 3)

    # Initialize cost model
    cmodel = FW_SimpleCostModel(max_width=10 * rng.random(),
                                sparse_inds=sparse_inds,
                                n_projs=rng.randrange(10, 100),
                                width_type='float128')

    # Check if correct
    for index in rng.sample(tuple(inds), k=len(inds)):
        # Get delta width
        delta_width_1 = cmodel.width(inds - {index}, dims) - cmodel.width(
            inds, dims)
        delta_width_2 = cmodel.delta_width(inds, dims, index)

        # Check if they're the same
        assert abs(delta_width_1 - delta_width_2) < 1e-6

        # Remove index from indices
        inds -= {index}


@pytest.mark.parametrize('seed', sample_seeds(1000))
def test_Probability(seed):

    def abs_error(x, y, atol=1e-5):
        return abs(x - y) < atol

    # Initialize rng
    rng = Random(seed)

    # Initialize
    base_prob = BaseProbability()

    # It should always be true
    assert all(
        map(
            lambda d, c: base_prob(d, c) == 1 and base_prob(d, c) == pickle.
            loads(pickle.dumps(base_prob))(d, c),
            mit.repeatfunc(rng.uniform, 1000, -100, 100),
            mit.repeatfunc(rng.uniform, 1000, -100, 100)))

    # Initialize
    greedy = Greedy()

    # Check greedy probability
    assert all(
        map(
            lambda d, c: greedy(d, c) ==
            (d <= 0) and greedy(d, c) == pickle.loads(pickle.dumps(greedy))
            (d, c), mit.repeatfunc(rng.uniform, 1000, -100, 100),
            mit.repeatfunc(rng.uniform, 1000, -100, 100)))

    # Check initialization
    assert all(
        map(lambda beta: MetropolisHastings(beta).beta == beta,
            mit.repeatfunc(rng.uniform, 1000, 0, 100)))

    # Check mh probability
    assert all(
        map(
            lambda beta, d, c: abs_error(
                MetropolisHastings(beta)(d, c), 1
                if d <= 0 else (1 + d / c)**-beta) and pickle.loads(
                    pickle.dumps(MetropolisHastings(beta))).beta == beta,
            mit.repeatfunc(rng.uniform, 1000, 0, 100),
            mit.repeatfunc(rng.uniform, 1000, -100, 100),
            mit.repeatfunc(rng.uniform, 1000, 0, 100)))

    def check_sa(mh, beta, delta_cost, old_cost):
        mh.beta = beta
        return abs_error(
            mh(delta_cost, old_cost), 1 if delta_cost <= 0 else
            (1 + delta_cost / old_cost)**-beta)

    # Check assignment of beta
    mh = MetropolisHastings()
    assert all(
        map(lambda beta, d, c: check_sa(mh, beta, d, c),
            mit.repeatfunc(rng.uniform, 1000, 0, 100),
            mit.repeatfunc(rng.uniform, 1000, -100, 100),
            mit.repeatfunc(rng.uniform, 1000, 0, 100)))

    # Check types
    for cost_type_ in ['float32', 'float64', 'float128', 'float1024']:
        assert MetropolisHastings(cost_type=cost_type_).cost_type == cost_type_


@pytest.mark.parametrize('seed', sample_seeds(200))
def test_all_close(seed: int, **kwargs):
    # Initialize rng
    rng = Random(seed)

    # Get params
    n_vars = kwargs.get('n_vars', rng.randrange(10_000, 20_000))

    # Get random array
    x = [rng.gauss(mu=0, sigma=1) for _ in range(n_vars)]
    assert all_close(x, x)

    # Flip a random element
    y = x[:]
    y[rng.randrange(n_vars)] *= -1
    assert not all_close(x, y)

    # Exponentiate arrays
    exp_x = list(map(exp, x))
    exp_y = list(map(exp, y))
    exp_x[rng.randrange(n_vars)] = 0
    assert all_logclose(exp_x, exp_x)
    assert not all_logclose(exp_x, exp_y)

    # Logclose should return always false for negative elements
    assert not all_logclose([1, -1, 1], [1, -1, 1])
