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

import itertools as its
from collections import Counter
from decimal import Decimal
from os import environ
from random import Random

import more_itertools as mit
import pytest
from tnco.ctree import ContractionTree
from tnco.optimize.finite_width import Optimizer as FW_Optimizer
from tnco.optimize.finite_width.cost_model import \
    SimpleCostModel as FW_SimpleCostModel
from tnco.optimize.infinite_memory import Optimizer as IM_Optimizer
from tnco.optimize.infinite_memory.cost_model import \
    SimpleCostModel as IM_SimpleCostModel
from tnco.optimize.prob import MetropolisHastings
from tnco.testing.utils import generate_random_tensors
from tnco.utils.tn import get_random_contraction_path

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


@pytest.mark.timeout(30)
@pytest.mark.usefixtures("timeout")
@repeat(10)
def test_InfiniteMemoryContraction(random_seed, **kwargs):
    # Get rng
    rng = Random(random_seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(50, 150))
    k = kwargs.get('k', rng.randint(2, 4))
    n_inds = kwargs.get('n_inds', rng.randint(100, 300))
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, n_inds // 5))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))
    disable_shared_inds = kwargs.get('disable_shared_inds',
                                     rng.choice([True, False]))
    cost_type = kwargs.get('cost_type',
                           'float128' if rng.randrange(2) else 'float1024')

    # Check minimum number of indices
    if (n_inds - n_output_inds) < n_tensors + 1 - k:
        pytest.skip("Too few indices")

    # Get tensors
    ts_inds, output_inds = generate_random_tensors(
        n_tensors=n_tensors,
        n_inds=n_inds,
        k=k,
        n_output_inds=n_output_inds,
        randomize_names=randomize_names,
        seed=random_seed)

    # Get random dimensions
    dims = dict(
        map(lambda x:
            (x, rng.randrange(1, 10)), mit.unique_everseen(
                mit.flatten(ts_inds)))) if rng.randrange(2) else rng.randrange(
                    1, 8)

    # Get sparse inds
    if rng.randint(0, 1):
        sparse_inds = None
    else:
        sparse_inds = list(mit.unique_everseen(mit.flatten(ts_inds)))
        sparse_inds = rng.sample(sparse_inds, k=len(sparse_inds) // 4)

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

    # Get optimizers
    opt = IM_Optimizer(ctree,
                       IM_SimpleCostModel(cost_type=cost_type),
                       seed=random_seed,
                       disable_shared_inds=disable_shared_inds)

    # Optimize (mh)
    mh = MetropolisHastings(cost_type=cost_type)
    beta = (0, 100)
    n_steps = 1000

    for i_ in range(n_steps):
        mh.beta = (beta[1] - beta[0]) * i_ / n_steps + beta[0]
        opt.update(mh)

    # Get hyper-count
    hyper_count = dict(
        its.starmap(lambda x, n: (x, n - 1),
                    Counter(mit.flatten(ts_inds)).items()))

    # Copy the inds
    ts_inds_ = ts_inds[:]

    # Convert dims
    dims_ = dict(
        zip(mit.unique_everseen(mit.flatten(ts_inds_)),
            its.repeat(dims))) if isinstance(dims, int) else dims

    def get_contraction_cost(x, y, z, dims):
        return IM_SimpleCostModel().contraction_cost(x, y, z, dims)

    # For each contraction
    total_cost = 0
    for x_, y_ in opt.min_ctree.path():
        # Get the corresponding inds
        x_, y_ = sorted((x_, y_))
        ys_ = frozenset(ts_inds_.pop(y_))
        xs_ = frozenset(ts_inds_.pop(x_))

        # Get shared inds
        sh_ = xs_ & ys_

        # Update the hyper-count
        for x_ in sh_:
            hyper_count[x_] -= 1
            assert hyper_count[x_] >= 0

        # Get the new inds
        zs_ = (xs_ ^ ys_) | frozenset(filter(lambda x: hyper_count[x],
                                             sh_)) | (output_inds & sh_)

        # Update total cost
        total_cost += get_contraction_cost(xs_, ys_, zs_, dims_)

        # Append the out
        ts_inds_.append(zs_)

    # No remaining inds to contract
    assert not sum(hyper_count.values())

    # Check last tensor
    assert len(ts_inds_) == 1 and ts_inds_[-1] == frozenset(output_inds)

    # Check total cost
    assert abs(Decimal(total_cost) -
               opt.min_total_cost) / opt.min_total_cost < 1e-2


@pytest.mark.timeout(30)
@pytest.mark.usefixtures("timeout")
@repeat(10)
def test_FiniteWidthContraction(random_seed, **kwargs):
    # Get rng
    rng = Random(random_seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(50, 150))
    k = kwargs.get('k', rng.randint(2, 4))
    n_inds = kwargs.get('n_inds', rng.randint(100, 300))
    n_output_inds = kwargs.get('n_output_inds', rng.randint(0, n_inds // 5))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))
    disable_shared_inds = kwargs.get('disable_shared_inds',
                                     rng.choice([True, False]))
    width_type = kwargs.get('width_type', 'float128')
    cost_type = kwargs.get('cost_type',
                           'float128' if rng.randrange(2) else 'float1024')

    # Check minimum number of indices
    if (n_inds - n_output_inds) < n_tensors + 1 - k:
        pytest.skip("Too few indices")

    # Get tensors
    ts_inds, output_inds = generate_random_tensors(
        n_tensors=n_tensors,
        n_inds=n_inds,
        k=k,
        n_output_inds=n_output_inds,
        randomize_names=randomize_names,
        seed=random_seed)

    # Get random dimensions
    dims = dict(
        map(lambda x:
            (x, rng.randrange(1, 10)), mit.unique_everseen(
                mit.flatten(ts_inds)))) if rng.randrange(2) else rng.randrange(
                    1, 8)

    # Get sparse inds
    if rng.randint(0, 1):
        sparse_inds = None
        n_projs = None
    else:
        sparse_inds = list(mit.unique_everseen(mit.flatten(ts_inds)))
        sparse_inds = rng.sample(sparse_inds, k=len(sparse_inds) // 4)
        n_projs = rng.randrange(1, 20)

    # Get slices to skip
    if rng.randint(0, 1):
        skip_slices = None
    else:
        skip_slices = list(mit.unique_everseen(mit.flatten(ts_inds)))
        skip_slices = rng.sample(skip_slices, k=len(skip_slices) // 10)

    def get_width(xs):
        return FW_SimpleCostModel(cost_type=cost_type,
                                  width_type=width_type,
                                  sparse_inds=sparse_inds,
                                  n_projs=n_projs,
                                  max_width=0).width(xs, dims)

    # Get max width (3-digits of precision)
    max_width = int(max(map(get_width, ts_inds)) * 1000) / 1000

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

    # Get optimizers
    try:
        opt = FW_Optimizer(ctree,
                           FW_SimpleCostModel(cost_type=cost_type,
                                              width_type=width_type,
                                              sparse_inds=sparse_inds,
                                              n_projs=n_projs,
                                              max_width=max_width),
                           skip_slices=skip_slices,
                           seed=random_seed,
                           disable_shared_inds=disable_shared_inds)

    # Skip test if too many indices are skipped
    except ValueError as e:
        if (str(e) == "Too many indices in 'skip_slices'."):
            pytest.skip("Too many indices in 'skip_slices'.")
        raise e

    # Optimize (mh)
    mh = MetropolisHastings(cost_type=cost_type)
    beta = (0, 100)
    n_steps = 1000

    for i_ in range(n_steps):
        mh.beta = (beta[1] - beta[0]) * i_ / n_steps + beta[0]
        opt.update(mh, update_slices=(i_ % 10 == 0))

        # Check that slices that are marked to be skipped are actually skipped
        assert not opt.slices & opt.skip_slices
        assert not opt.min_slices & opt.skip_slices

    # Get hyper-count
    hyper_count = dict(
        its.starmap(lambda x, n: (x, n - 1),
                    Counter(mit.flatten(ts_inds)).items()))

    # Copy the inds
    ts_inds_ = ts_inds[:]

    # Convert dims
    dims_ = dict(
        zip(mit.unique_everseen(mit.flatten(ts_inds_)),
            its.repeat(dims))) if isinstance(dims, int) else dims

    def get_contraction_cost(x, y, z, dims, slices):
        return FW_SimpleCostModel(cost_type=cost_type,
                                  width_type=width_type,
                                  sparse_inds=sparse_inds,
                                  n_projs=n_projs,
                                  max_width=max_width).contraction_cost(
                                      x, y, z, dims, slices)

    # For each contraction
    total_cost = 0
    for x_, y_ in opt.min_ctree.path():
        # Get the corresponding inds
        x_, y_ = sorted((x_, y_))
        ys_ = frozenset(ts_inds_.pop(y_))
        xs_ = frozenset(ts_inds_.pop(x_))

        # Get shared inds
        sh_ = xs_ & ys_

        # Update the hyper-count
        for x_ in sh_:
            hyper_count[x_] -= 1
            assert hyper_count[x_] >= 0

        # Get the new inds
        zs_ = (xs_ ^ ys_) | frozenset(filter(lambda x: hyper_count[x],
                                             sh_)) | (output_inds & sh_)

        # Check width
        assert get_width(zs_ - opt.min_slices) <= max_width

        # Update total cost
        total_cost += get_contraction_cost(xs_, ys_, zs_, dims_, opt.min_slices)

        # Append the out
        ts_inds_.append(zs_)

    # No remaining inds to contract
    assert not sum(hyper_count.values())

    # Check last tensor
    assert len(ts_inds_) == 1 and ts_inds_[-1] == frozenset(output_inds)

    # Check total cost
    assert abs(Decimal(str(total_cost)) -
               opt.min_total_cost) / opt.min_total_cost < 1e-2
