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
from collections import defaultdict
from random import Random
from string import ascii_letters
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union

import more_itertools as mit
from rich.console import Console
from rich.progress import Progress, track

from tnco.ctree import ContractionTree
from tnco.ordered_frozenset import OrderedFrozenSet
from tnco.typing import Index, IndexName

__all__ = [
    'get_connected_components', 'generate_random_inds',
    'generate_random_tensors', 'is_valid_contraction_tree'
]


def get_connected_components(ts_inds: Iterable[List[Index]],
                             *,
                             verbose: Optional[int] = False) -> List[List[int]]:
    """Get connected components.

    Given a list of tensor's indices, return the connected components.

    Args:
        ts_inds: List of indices, with each item being the indices of a tensor.

    Returns:
        List where Each element corresponds to a list of positions of tensors
        belonging to the same connected component.
    """
    # Convert to list, and ignore empty tensors
    ts_inds = list(filter(len, ts_inds))

    # If empty, return empty
    if not len(ts_inds):
        return []

    # Initialize colors
    index2color = {}
    color2index = {}

    for inds_ in track(ts_inds,
                       description="Finding CC...",
                       console=Console(stderr=True),
                       disable=(verbose <= 0)):

        # If empty, just skip it
        if not len(inds_):
            continue

        # Get all colors associated to the indices in the tensor
        all_colors_ = tuple(
            mit.unique_everseen(
                filter(lambda x: x is not None,
                       map(lambda x: index2color.get(x), inds_))))

        # Get all inds
        all_inds_ = fts.reduce(
            op.or_, map(lambda x: color2index.get(x, frozenset()), all_colors_),
            frozenset(inds_))

        # Get color
        color_ = min(all_colors_) if all_colors_ else max([-1, *color2index
                                                          ]) + 1

        # Update colors
        index2color.update(zip(all_inds_, its.repeat(color_)))
        mit.consume(map(lambda x: color2index.pop(x, None), all_colors_))
        color2index[color_] = all_inds_

    # Check colors
    assert list(
        mit.unique_everseen(
            map(
                lambda inds: mit.ilen(
                    mit.unique_everseen(map(index2color.get, inds))),
                filter(len, ts_inds)))) == [1]

    # Check all indices
    assert all(
        map(
            lambda x: x == 2,
            mit.map_reduce(
                its.chain(mit.unique_everseen(mit.flatten(ts_inds)),
                          index2color), lambda x: x, lambda x: 1,
                sum).values()))

    # Assign a new color to all the empty tensors
    def new_color():
        c_ = max(color2index) + 1
        while 1:
            yield c_
            c_ += 1

    new_color = new_color()

    # Get connected components
    cc = (map(
        lambda inds: index2color[mit.first(inds)]
        if len(inds) else next(new_color), ts_inds))

    # Group indices in the same cc together
    return list(
        mit.map_reduce(enumerate(cc), lambda x: x[1], lambda x: x[0],
                       tuple).values())


def generate_random_inds(n: int, *, seed: Optional[int] = None) -> List[Index]:
    """Generate unique random names.

    Generate unique random names.

    Args:
        n: The number of unique random names to generate.
        seed: Seed to use.

    Returns:
        List of random names.
    """
    # Return if 'n' is trivial
    if n <= 0:
        return ()

    # Initialize rng
    rng = Random(seed)

    def get_():
        if ((r := rng.randrange(3))) == 0:
            return ''.join(rng.choices(ascii_letters, k=12))
        elif r == 1:
            return tuple(map(rng.randrange, its.repeat(2**32, 4)))
        elif r == 2:
            return (''.join(rng.choices(ascii_letters,
                                        k=12)), rng.randrange(2**32))
        else:
            raise NotImplementedError()

    # Generate random names, and check for uniqueness
    names_ = OrderedFrozenSet()
    while len(names_ := names_.union([get_()])) != n:
        pass

    # Return names
    return tuple(names_)


def generate_random_tensors(
    n_tensors: int,
    n_inds: int,
    k: Optional[int] = 2,
    *,
    n_output_inds: Optional[int] = 0,
    n_cc: Optional[int] = 1,
    randomize_names: Optional[bool] = True,
    seed: Optional[int] = None,
    verbose: Optional[int] = False
) -> Tuple[List[List[IndexName]], FrozenSet[IndexName]]:
    """Generate random tensors.

    Generate random tensors.

    Args:
        n_tensors: The total number of tensors to generate will be
            'n_tensors * n_cc'.
        n_inds: The total number of indices to generate will be
            'n_inds * n_cc'.
        k: int, Number of tensors an index connects.
        n_output_inds: The total number of output indices will be
            'n_output_inds * n_cc'.
        n_cc: Number of connected components.
        randomize_names: If 'True', random names for indices are used.
        seed: Seed to use.
        verbose: Verbose output.

    Returns:
        A tuple of random tensors and output indices.
    """
    # Initialize RNG
    rng = Random(seed)

    # Number of hyper/non-hyper output_inds
    n_hyper_output_inds = 0 if k <= 2 else \
                            n_output_inds // 2 + n_output_inds % 2
    n_nhyper_output_inds = n_output_inds if k <= 2 else n_output_inds // 2
    assert (n_hyper_output_inds + n_nhyper_output_inds) == n_output_inds

    def permutation(x):
        x = list(x)
        return rng.sample(x, k=len(x))

    # Check minimum number of indices
    if (n_inds - n_output_inds) < n_tensors + 1 - k:
        raise ValueError("Too few indices.")

    # Initialize
    all_tensors = []
    all_output_inds = []

    # Generate single cc
    for _ in track(range(n_cc),
                   console=Console(stderr=True),
                   disable=(verbose <= 0),
                   description="Generating CC..."):

        for _ in range(10):

            # Generate the connected component first
            avail_tensors: Set[int] = set(range(n_tensors))
            used_tensors: Set[int] = set()
            inds: List[List[int]] = []

            inds.append(rng.sample(list(avail_tensors), k=k))
            avail_tensors -= set(inds[-1])
            used_tensors |= set(inds[-1])
            for _ in range(n_tensors - k):
                x_ = rng.sample(list(avail_tensors), k=1)[0]
                inds.append(rng.sample(list(used_tensors), k=k - 1) + [x_])
                avail_tensors -= {x_}
                used_tensors |= {x_}

            # All tensors should be present
            assert used_tensors == set(
                range(n_tensors)) and not avail_tensors and set(
                    mit.flatten(inds)) == set(range(n_tensors))

            # Check number of inds
            assert len(inds) == n_tensors + 1 - k

            # Add the remaining inds (escluding output inds)
            inds.extend(
                rng.sample(range(n_tensors), k=k)
                for _ in range(n_inds - len(inds) - n_output_inds))

            # Add output indices with a dedicated index
            inds.extend([rng.randrange(n_tensors), '*']
                        for _ in range(n_nhyper_output_inds))

            # Add output hyper-indices
            inds.extend(
                rng.sample(range(n_tensors), k=k - 1) + ['*']
                for _ in range(n_hyper_output_inds))

            # Check
            assert len(inds) == n_inds

            # Get tensors from inds
            tensors = defaultdict(list)

            for x_, ts_ in enumerate(inds):
                for t_ in ts_:
                    tensors[t_].append(x_)

            # Check number of output inds
            assert len(tensors['*']) == n_output_inds

            # Get output inds
            output_inds = frozenset(tensors.pop('*'))

            # Get tensors
            tensors = list(map(tuple, map(permutation, tensors.values())))

            # Update
            if mit.ilen(mit.unique_everseen(tensors)) == n_tensors:
                all_tensors.append(tensors)
                all_output_inds.append(output_inds)
                break

        # Just give up
        else:
            raise ValueError("Cannot generate random tensors.")

    # Combine the tensors
    all_tensors = fts.reduce(
        op.add,
        its.starmap(
            lambda n, tensors: list(
                map(lambda xs: tuple(map(lambda x: x + n * n_inds, xs)), tensors
                   )), enumerate(all_tensors)))
    rng.shuffle(all_tensors)

    # Combine all output_inds
    all_output_inds = fts.reduce(
        op.or_,
        its.starmap(
            lambda n, output_inds: frozenset(x + n * n_inds
                                             for x in output_inds),
            enumerate(all_output_inds)))

    # Randomize names of inds
    if randomize_names:
        with Progress(disable=(verbose <= 0),
                      console=Console(stderr=True)) as pbar:
            task = pbar.add_task("Randomizing names...", total=3)

            # Initialize map
            all_inds_ = OrderedFrozenSet(mit.flatten(all_tensors))
            inds_map_ = dict(
                zip(
                    all_inds_,
                    generate_random_inds(len(all_inds_),
                                         seed=rng.randrange(2**32))))
            pbar.update(task, advance=1)

            # Update names
            all_tensors = list(
                map(lambda xs: tuple(map(inds_map_.get, xs)), all_tensors))
            pbar.update(task, advance=1)

            all_output_inds = frozenset(map(inds_map_.get, all_output_inds))
            pbar.update(task, advance=1)

            pbar.update(task, refresh=True)

    # All tensors should be unique
    assert mit.ilen(mit.unique_everseen(all_tensors)) == len(all_tensors)

    return all_tensors, all_output_inds


def is_valid_contraction_tree(ctree: ContractionTree,
                              ts_inds: Optional[Iterable[List[Index]]] = None,
                              dims: Optional[Union[Dict[Index, int],
                                                   int]] = None,
                              output_inds: Optional[Iterable[Index]] = None,
                              hyper_count: Optional[Dict[Index, int]] = None,
                              check_shared_inds: Optional[bool] = None) -> bool:
    """Check if a 'ContractionTree' is valid.

    Check if a 'ContractionTree' is valid.

    Args:
        ctree: Contraction tree to check.
        ts_inds: Tensors to check against the contraction tree.
        dims: Dimensions to check against the contraction tree.
        output_inds: Output indices to check against the contraction tree.
        hyper_count: Expected hyper-count for each index.
        check_shared_inds: If 'True', check that every pair of contracted
            tensors share at least one index.

    Returns:
        'True' if the contraction tree is valid, othewise 'False'.
    """
    # Check pickle
    copy_ctree = pickle.loads(pickle.dumps(ctree))
    if copy_ctree != ctree:
        return False

    # Updating the copy shouldn't affect the original
    for i_ in range(len(copy_ctree)):
        copy_ctree.swap_with_nn(i_)
    assert copy_ctree != ctree

    # Check dimensions
    if dims is not None and not all(
            its.starmap(
                lambda x, d: (dims if isinstance(dims, int) else dims[x]) == d,
                ctree.dims.items())):
        return False

    # Check inds for leaves
    if ts_inds is not None:
        if ctree.inds[:ctree.n_leaves] != tuple(map(frozenset, ts_inds)):
            return False
        if frozenset(mit.flatten(ts_inds)) != ctree.all_inds():
            return False

    # Check if contractions are valid
    if not ctree.is_valid(check_shared_inds=False if check_shared_inds is
                          None else check_shared_inds):
        return False

    # Check output inds
    if output_inds is not None and (ctree.inds[-1] != frozenset(output_inds) or
                                    ctree.output_inds()
                                    != frozenset(output_inds)):
        return False

    # Check hyper-count
    if hyper_count is not None:
        # Count how many times an index is contracted
        hyper_count_from_ct = defaultdict(int)
        for node_ in ctree.nodes:
            if not node_.is_leaf():
                ix_, iy_ = ctree.inds[node_.children[0]], ctree.inds[
                    node_.children[1]]
                if check_shared_inds and not (ix_ & iy_):
                    return False

                for is_ in ix_ & iy_:
                    hyper_count_from_ct[is_] += 1

        # Increment by one for all the output inds
        for x_ in ctree.inds[-1]:
            hyper_count_from_ct[x_] += 1

        # Check hyper-inds
        if not all(
                its.starmap(lambda x, v: hyper_count[x] == v,
                            hyper_count_from_ct.items())):
            return False

    # Everything, OK
    return True
