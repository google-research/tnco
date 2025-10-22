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

from typing import (Iterable, Optional, List, Tuple, Union, Dict, FrozenSet)
from tnco.ordered_frozenset import OrderedFrozenSet
from tnco.typing import Index, TensorName, Array
from collections import Counter, defaultdict
from rich.progress import Progress, track
import tnco.utils.tensor as tensor_utils
from rich.console import Console
import more_itertools as mit
from random import Random
import functools as fts
import itertools as its
import operator as op
import autoray as ar
import math

__all__ = [
    'get_random_contraction_path', 'read_inds', 'fuse', 'decompose_hyper_inds',
    'merge_contraction_paths', 'contract'
]


def get_random_contraction_path(
        ts_inds: Iterable[List[Index]],
        output_inds: Optional[Iterable[Index]] = None,
        *,
        merge_paths: Optional[bool] = True,
        autocomplete: Optional[bool] = True,
        seed: Optional[int] = None,
        verbose: Optional[int] = False,
        **kwargs) -> Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]]:
    """Generate random contraction.

    Generate a random contraction path from 'ts_inds'.

    Args:
        ts_inds: List of indices, with each item corresponding the indices of a
            tensor.
        output_inds: List of output indices. Must be provided if 'ts_inds' has
            hyper-indices.
        merge_paths: If 'True' merge all tensors regardless if they belong to
            different connected components. If 'False', return a contraction
            path for each connected components.
        autocomplete: If 'merge_paths=True' and 'autocomplete=True', all the
            tensors belonging to different connected components are contracted
            to a single tensor. Otherwise, if 'merge_paths=True' and
            'autocomplete=False', tensors belonging to different connected
            components are not contracted, resulting to an incomplete path.
        seed: Seed to use.
        verbose: Verbose output.

    Returns:
        If 'merge_path=True', return a single path in the SSA format.
        Otherwise, return multuple paths in the SSA format for each connected
        component. Each path guarantees that only tensors that share at least
        one index are included in the path.

    Raises:
        ValueError: If arguments are not consistent with each other.
    """
    # Extra args
    _return_contraction = kwargs.pop('_return_contraction', False)
    if kwargs:
        raise TypeError("Got an expected keyword argument(s).")

    # Convert inds to frozen sets
    ts_inds = list(map(OrderedFrozenSet, ts_inds))

    # If empty, just return
    if not ts_inds:
        return []

    # Get hyper-count
    hyper_count = dict(
        its.starmap(lambda x, n: (x, n - 1),
                    Counter(mit.flatten(ts_inds)).items()))

    if output_inds is None:
        # Raise an error if there are hyper-inds but output_inds has not been
        # provided
        if any(map(lambda x: x > 1, hyper_count.values())):
            raise ValueError(
                "'output_inds' must be provided if 'ts_inds' has hyper-indices."
            )

        # Otherwise, generate output inds from hyper_count
        output_inds = map(op.itemgetter(0),
                          filter(lambda x: x[1] == 0, hyper_count.items()))

    # Convert to set
    output_inds = frozenset(output_inds)

    # Check output indices
    if not output_inds.issubset(mit.flatten(ts_inds)):
        raise ValueError("'output_inds' is not consistent with 'ts_inds'.")

    # Increment hyper-count for hyper-inds
    for x_ in output_inds:
        hyper_count[x_] += 1

    # Get map of # For each index, get the positions to the attached tensors
    index2pos = mit.map_reduce(
        mit.flatten(
            its.starmap(lambda p, xs: zip(xs, its.repeat(p)),
                        enumerate(ts_inds))), op.itemgetter(0),
        op.itemgetter(1), frozenset)

    # Get adjacency matrix
    adj = dict(
        its.starmap(
            lambda p, xs: (p,
                           frozenset(
                               filter(lambda q: q != p,
                                      mit.flatten(map(index2pos.get, xs))))),
            enumerate(ts_inds)))

    # Ignore all c-numbers
    adj = dict(filter(lambda x: len(x[1]), adj.items()))

    # Initialize available tensors
    avail_tensors = list(adj)

    # Initialize current tensor to contract
    px = None

    # Initialize rng
    rng = Random(seed)

    # Get the initial number of tensors
    n_tensors = len(ts_inds)

    # Initialize contraction
    contraction = []

    # While there are available tensors ...
    with Progress(disable=(verbose <= 0), console=Console(stderr=True)) as pbar:
        total_pbar = len(avail_tensors)
        task = pbar.add_task("Getting contraction path...", total=total_pbar)

        while avail_tensors:
            if px is None or not adj[px]:
                # Get a random available posistion
                px = avail_tensors.pop(rng.randrange(len(avail_tensors)))

                # Reset inds
                ts_inds = ts_inds[:n_tensors]

                # Add a new contraction
                contraction.append([])

            # Get a random tensor among the adjacent to py, maximizing the
            # number of shared inds
            py = rng.choice(
                sorted(mit.map_reduce(
                    map(lambda py:
                        (len(ts_inds[px] & ts_inds[py]), py), adj[px]),
                    op.itemgetter(0), op.itemgetter(1), list).items(),
                       reverse=True)[0][1])

            # Remove py from the available tensors
            avail_tensors.pop(avail_tensors.index(py))

            # Get inds
            xs, ys = ts_inds[px], ts_inds[py]

            # Get shared inds
            shared = xs & ys

            # They should always share an index
            assert len(shared)

            # Update hyper-count to all shared inds
            for x_ in shared:
                hyper_count[x_] -= 1
                assert hyper_count[x_] >= 0

            # Get all hyper-inds
            hyper = frozenset(
                map(
                    op.itemgetter(0),
                    filter(lambda x: x[1],
                           zip(shared, map(hyper_count.get, shared)))))

            # Get the new inds
            pz, zs = len(ts_inds), (xs ^ ys) | hyper

            # Update contraction
            ts_inds.append(zs)
            contraction[-1].append((px, py, pz))

            # Update adj matrix
            adj[pz] = (adj.pop(px) | adj.pop(py)) - {px, py}
            for p_ in adj[pz]:
                adj[p_] -= {px, py}
                adj[p_] |= {pz}

            # Always use the last one
            px = pz

            # Update progressbar
            pbar.update(task, completed=total_pbar - len(avail_tensors))

        # Last update of the progressbar
        pbar.update(task, completed=total_pbar, refresh=True)

        # Check output inds
        assert ts_inds[-1].issubset(output_inds)

    # For testing only
    if _return_contraction:
        return contraction

    # Normalize contractions
    ssa_paths = []
    for cntr in contraction:
        # Get all positions
        pos = list(range(max(mit.flatten(cntr)) + 1))

        # Get SSA
        ssa_paths.append([])
        for px, py, _ in cntr:
            px, py = sorted((px, py))
            pos.pop(py := pos.index(py))
            pos.pop(px := pos.index(px))
            ssa_paths[-1].append((px, py))

    # Return paths
    return merge_contraction_paths(
        n_tensors, ssa_paths,
        autocomplete=autocomplete) if merge_paths else ssa_paths


def merge_contraction_paths(
        n_tensors: int,
        paths: Iterable[List[Tuple[int, int]]],
        *,
        autocomplete: Optional[bool] = True) -> List[Tuple[int, int]]:
    """Merge contraction paths.

    Merge contraction paths for disconnected tensor networks.

    Args:
        n_tensors: Number of total tensors
        paths: Contraction paths to merge in the SSA format.
        autocomplete: If 'True', the merged path will include the contraction
            of the disconnected tensors.

    Returns:
        Merged path in the SSA format.

    Raises:
        ValueError: If 'paths' are not valid or not disconnected.

    Notes:
        See 'tnco.utils.tn.get_random_contraction_path'.
    """
    # Initialize merged path and positions
    merged_pos = list(range(n_tensors))
    merged_path = []

    # For each path ...
    for i, path in enumerate(paths):

        # Initialize positions
        pos = list(range(n_tensors))

        # "contract"
        for x, y in path:

            # Get contracted indices
            x, y = sorted((x, y))
            y = pos.pop(y)
            x = pos.pop(x)
            pos.append((i, len(pos)))

            # Update merged path
            try:
                mx, my = sorted((merged_pos.index(x), merged_pos.index(y)))
            except ValueError:
                raise ValueError("'paths' are not valid or not disconnected.")
            merged_path.append((mx, my))
            merged_pos.pop(my)
            merged_pos.pop(mx)
            merged_pos.append(pos[-1])

    # Add the remaining contraction
    if autocomplete:
        merged_path += [(0, 1)] * (len(merged_pos) - 1)

    # Return merged path
    return merged_path


def read_inds(
    inds_map: Dict[Index, Tuple[int, TensorName]],
    *,
    output_index_token: TensorName = '*',
    sparse_index_token: TensorName = '/'
) -> Tuple[Dict[TensorName, Tuple[Index, ...]], Dict[Index, int],
           FrozenSet[Index], FrozenSet[Index]]:
    """Read indices.

    Convert list of indices to a map of tensors.

    Args:
        inds_map: Map of indices in the format:
                '{index_1: (dim, tname_1, tname_2, ...), ...}'.
        output_index_token: The token to use to identify output inds.
        sparse_index_token: The token to use to identify sparse inds.

    Returns:
        It returns the loaded tensors as a tuple with the following format:
            tensor_map: Map of tensors. Each tensors is a list of indices
                attached to it.
            dims: Dimension of each index.
            output_inds: Set of output indices.
            sparse_inds: Set of sparse indices.

    Raises:
        ValueError: If 'output_index_token' and 'sparse_index_token' are the
            same.
    """
    # Check tokens
    if output_index_token == sparse_index_token:
        raise ValueError(
            "'output_index_token' and 'sparse_index_token' must differ.")

    # Build map of tensors
    tensor_map = defaultdict(list)
    dims = {}

    for i_, (d_, *ts_) in inds_map.items():
        dims[i_] = int(d_)
        for t_ in ts_:
            tensor_map[t_].append(i_)

    # Get output inds
    output_inds = frozenset(tensor_map.pop(output_index_token, ()))

    # Get sparse inds
    sparse_inds = frozenset(tensor_map.pop(sparse_index_token, ()))

    # Return
    return dict(zip(tensor_map,
                    map(tuple,
                        tensor_map.values()))), dims, output_inds, sparse_inds


def fuse(
    ts_inds: Iterable[List[Index]],
    dims: Union[int, Dict[Index, int]],
    max_width: float,
    output_inds: Optional[Iterable[Index]] = None,
    *,
    exclude_inds: Optional[Iterable[Index]] = (),
    seed: Optional[int] = None,
    return_fused_inds: Optional[bool] = False,
    verbose: Optional[int] = False
) -> Tuple[List[Tuple[int, int]], Optional[List[Tuple[Index]]]]:
    """Fuse indices.

    Fuse 'ts_inds' up to 'max_width'.

    Args:
        ts_inds: List of indices, with each item being the indices of a tensor.
        dims: Dimension of each index.
        max_width: Maximum width to use. The width is defined as sum of the
            logarithms of all the dimensions of a given tensor.  Tensors are
            contracted so that the width of the contracted tensor is smaller
            than 'max_width'.
        output_inds: List of output indices. Must be provided if 'tensor' has
            hyper-indices.
        exclude_inds: Indices that must not be contracted.
        seed: Seed to use.
        return_fused_inds: If 'True', return the fused indices.
        verbose: Verbose output.

    Returns:
        The contraction path in the SSA format. If 'return_fused_inds=True',
        also return the corresponding indices of the fused tensors.

    Raises:
        ValueError: If arguments are not consistent with each other.
    """

    # Initialize rng
    rng = Random(seed)

    # Copy tensors
    ts_inds = dict(enumerate(map(tuple, ts_inds)))

    # Get all inds
    all_tensors_inds = OrderedFrozenSet(
        mit.unique_everseen(mit.flatten(ts_inds.values())))

    # Conver to frozenset
    exclude_inds = frozenset(exclude_inds)
    if not exclude_inds.issubset(all_tensors_inds):
        raise ValueError("'exclude_inds' contains indices not in 'ts_inds'.")

    # Convert dims to dict
    try:
        dims = dict(zip(all_tensors_inds, its.repeat(int(dims))))
    except (TypeError, ValueError):
        dims = dict(dims)
    if not all_tensors_inds.issubset(dims):
        raise ValueError("'dims' is missing some indices.")

    # How to compute the width
    def get_width(xs):
        return sum(map(math.log2, map(dims.get, xs)))

    # Get hyper-count
    hyper_count = dict(
        its.starmap(lambda x, n: (x, n - 1),
                    Counter(mit.flatten(ts_inds.values())).items()))

    if output_inds is None:
        # Raise an error if there are hyper-inds but output_inds has not been
        # provided
        if any(map(lambda x: x > 1, hyper_count.values())):
            raise ValueError(
                "'output_inds' must be provided if 'ts_inds' has hyper-indices."
            )

        # Otherwise, generate output inds from hyper_count
        output_inds = map(op.itemgetter(0),
                          filter(lambda x: x[1] == 0, hyper_count.items()))

    # Convert to set
    output_inds = frozenset(output_inds)

    # Check output indices
    if not output_inds.issubset(mit.flatten(ts_inds.values())):
        raise ValueError("'output_inds' is not consistent with 'ts_inds'.")

    # Build adjacency matrix
    index2tensors = dict(
        map(
            lambda xs: (xs[0][1], set(map(op.itemgetter(0), xs))),
            mit.map_reduce(
                mit.flatten(
                    its.starmap(lambda x, ys: zip(its.repeat(x), ys),
                                enumerate(ts_inds.values()))),
                op.itemgetter(1)).values()))

    # Initialize available inds
    avail_inds = list(all_tensors_inds - exclude_inds - frozenset(
        map(op.itemgetter(0), filter(lambda x: x[1] == 0, hyper_count.items())))
                     )

    # Initialize tensor idx
    t_idx = len(ts_inds)

    # All merged inds
    all_merged_inds = set()

    # All merged tensors
    all_merged_tensors = []

    # While there are available indices to contract
    with Progress(disable=(verbose <= 0), console=Console(stderr=True)) as pbar:
        total_pbar = len(avail_inds)
        task = pbar.add_task("Fusing tensors...", total=total_pbar)

        while avail_inds:
            # Select random position
            index = avail_inds.pop(rng.randrange(len(avail_inds)))

            # Skip if the index has been already contracted
            if not hyper_count.get(index):
                #print('## skip')
                continue

            # Get tensors
            px, py = rng.sample(tuple(index2tensors[index]), k=2)
            tx, ty = ts_inds[px], ts_inds[py]

            # Get all indices
            all_inds = frozenset(tx) | frozenset(ty)

            # If there is at least one excluded index, skip fusion
            if all_inds & exclude_inds:
                continue

            # Get shared inds
            shared_inds = frozenset(tx) & frozenset(ty)

            # They should share at least one index
            assert index in shared_inds

            # All shared inds must have a hyper-count > 0
            assert all(
                map(lambda x: x[0] not in shared_inds or x[1] > 0,
                    hyper_count.items()))

            # Get hyper inds
            hyper_inds = frozenset(
                map(
                    op.itemgetter(0),
                    filter(lambda x: x[0] in shared_inds and x[1] > 1,
                           hyper_count.items())))

            # Generate new inds
            tz = (frozenset(tx) ^ frozenset(ty)) | hyper_inds | (output_inds &
                                                                 all_inds)

            # Sort new inds as they appear in tx and ty
            tz = tuple(
                mit.unique_everseen(
                    its.chain(filter(lambda x: x in tz, tx),
                              filter(lambda y: y in tz, ty))))

            # If width is larger than provided, do nothing
            if get_width(tz) > max_width:
                continue

            # Update hyper-count
            for x in shared_inds:
                hyper_count[x] -= 1

            # For all the output inds, remove tx and ty, and add tz
            for x in tz:
                index2tensors[x] -= {px, py}
                index2tensors[x] |= {t_idx}

            # For all the shared, remove the index
            for x in (shared_inds - hyper_inds - output_inds):
                del index2tensors[x]

            # Update merged inds
            all_merged_inds |= shared_inds

            # Remove old tensors, and add new one
            del ts_inds[px]
            del ts_inds[py]
            ts_inds[t_idx] = tz
            t_idx += 1

            # If hyper-count is larger than zero, add back to available
            if hyper_count.get(index):
                avail_inds.append(index)

            # Update merged tensors
            all_merged_tensors.append((px, py, tz))

            # Update pbar
            pbar.update(task, completed=total_pbar - len(avail_inds))

        # Final update
        pbar.update(task, completed=total_pbar, refresh=True)

    # No excluded index should appear in merged
    assert not all_merged_inds & exclude_inds

    # All hyper counts should be positive
    assert all(map(lambda x: x >= 0, hyper_count.values()))

    # Renormalize contraction
    path = []
    fused_inds = []
    positions = list(range(t_idx))
    for px, py, tz in all_merged_tensors:
        px, py = sorted((px, py))
        del positions[(py := positions.index(py))]
        del positions[(px := positions.index(px))]
        path.append((px, py))
        fused_inds.append(tz)

    # Return the contraction to fuse the tensors
    return (path, fused_inds) if return_fused_inds else path


def decompose_hyper_inds(
    arrays: Iterable[Array],
    ts_inds: Iterable[List[Index]],
    *,
    atol: float = 1e-8
) -> Tuple[List[Array], List[List[Index]], Dict[Index, Index]]:
    """Decompose 'arrays' in hyper-indices.

    Decompose 'arrays' in hyper-indices.

    Args:
        arrays: List of array representing the tensor network to decompose.
        ts_inds: List of indices, with each item corresponding the indices of a
            tensor.
        atol: Absolute tollerance when checking for hyper-indices.

    Returns:
        It returns the decomposed arrays using the format:
            arrays: The list of new arrays after decomposing the hyper inds.
            ts_inds: The list of new indices after decomposing the hyper inds.
            hyper_inds_map:Map of hyper-indices, with `hyper_inds_map[x] == y`
                meaning that the index `x` is now called `y` after the
                decomposition.
    """
    # Get all available inds
    ts_inds = list(ts_inds)
    all_inds = OrderedFrozenSet(mit.flatten(ts_inds))

    # New tn
    new_arrays = []
    new_ts_inds = []

    # Map of new merged inds
    new_hyper_inds = []

    # Get all decomposition
    for array, inds in zip(arrays, ts_inds):
        (new_array,
         new_inds), hyper_inds = tensor_utils.decompose_hyper_inds(array,
                                                                   inds,
                                                                   atol=atol)
        new_arrays.append(new_array)
        new_ts_inds.append(new_inds)
        new_hyper_inds.append(hyper_inds)

    # Initialize colormap
    index2color = dict(zip(all_inds, its.count()))
    color2inds = dict(
        its.starmap(lambda x, c: (c, OrderedFrozenSet([x])),
                    index2color.items()))

    # Get clusters
    for hyper_x, xs in zip(
            mit.flatten(map(lambda x: x.keys(), new_hyper_inds)),
            mit.flatten(map(lambda x: x.values(), new_hyper_inds))):
        if len(xs):
            # Get all inds
            xs = frozenset(xs).union([hyper_x])

            # Get all colors
            cs = sorted(mit.unique_everseen(map(index2color.get, xs)))

            # Merge all inds
            color2inds[cs[0]] = fts.reduce(op.or_, map(color2inds.pop, cs))

            # Update colors
            index2color.update(dict(zip(color2inds[cs[0]], its.repeat(cs[0]))))

    # Get new map of inds
    hyper_inds_map = dict(
        mit.flatten(
            map(lambda xs: zip(xs, its.repeat(mit.first(xs))),
                color2inds.values())))

    # Update the new inds accordingly
    new_ts_inds = list(
        map(lambda xs: tuple(map(hyper_inds_map.get, xs)), new_ts_inds))

    # Return the decomposed tensors
    return new_arrays, new_ts_inds, hyper_inds_map


def contract(
    path: Iterable[Tuple[int, int]],
    ts_inds: Iterable[List[Index]],
    output_inds: Optional[Iterable[Index]] = None,
    arrays: Iterable[Array] = None,
    dims: Union[int, Dict[Index, int]] = None,
    *,
    backend: Optional[str] = None,
    verbose: Optional[int] = False
) -> Tuple[List[List[Index]], FrozenSet[Index], Optional[List[Array]]]:
    """Contract tensor network.

    Contract tensor network following 'path'.

    Args:
        path: Path to follow for the contraction in the SSA format.
        ts_inds: Indices associated to 'arrays'.
        output_inds: Output indices (optional if 'ts_inds' does not have
            hyper-indices).
        arrays: If provided, the arrays to contract.
        dims: Dimensions of each index. Must be provided if 'arrays' is not
            provided.
        backend: Backend to use for the contraction. See: `autoray.do`.
        verbose: Verbose output.

    Returns:
        It returns the result of the contraction as a tuple of list of indices
        (for each resulting tensor), and the final output indices. If 'arrays'
        is provided, it also return the contracted arrays.

    Raises:
        ValueError: If 'path' is not valid.
        ValueError: If arguments for the tensor network are not consistent with
            each other.
    """
    # Use specific backend
    do = fts.partial(ar.do, like=backend)

    # Either 'dims' or 'arrays' must be provided
    if dims is None and arrays is None:
        raise ValueError("Either 'dims' or 'arrays' must be provided.")

    # Get indices
    ts_inds = list(map(tuple, ts_inds))

    # Convert int to dict
    if dims is not None:
        try:
            dims = dict(zip(mit.flatten(ts_inds), its.repeat(int(dims))))
        except (ValueError, TypeError):
            pass

    # Get arrays
    arrays = None if arrays is None else list(
        map(fts.partial(do, 'asarray'), arrays))

    # Get dimensions
    if arrays is None:
        if not frozenset(dims).issuperset(mit.flatten(ts_inds)):
            raise ValueError("'ts_inds' has indices not in 'dims'.")
    else:
        dims_ = dict(
            mit.flatten(map(lambda a, xs: zip(xs, a.shape), arrays, ts_inds)))
        if len(arrays) != len(ts_inds) or not all(
                map(lambda a, xs: tuple(a.shape) == tuple(map(dims_.get, xs)),
                    arrays, ts_inds)):
            raise ValueError("'ts_inds' is not consistent with 'arrays'.")

        # Check with dims, if provided
        if dims is None:
            dims = dims_
        else:
            if not all(its.starmap(lambda x, d: dims[x] == d, dims_.items())):
                raise ValueError("'dims' and 'arrays' are not compatible.")

    # Get hyper-count
    hyper_count = dict(
        its.starmap(lambda x, n: (x, n - 1),
                    Counter(mit.flatten(ts_inds)).items()))

    if output_inds is None:
        # Raise an error if there are hyper-inds but output_inds has not been
        # provided
        if any(map(lambda x: x > 1, hyper_count.values())):
            raise ValueError(
                "'output_inds' must be provided if 'ts_inds' has hyper-indices."
            )

        # Otherwise, generate output inds from hyper_count
        output_inds = map(op.itemgetter(0),
                          filter(lambda x: x[1] == 0, hyper_count.items()))

    # Convert to set
    output_inds = frozenset(output_inds)

    # Check output indices
    if not output_inds.issubset(mit.flatten(ts_inds)):
        raise ValueError("'output_inds' is not consistent with 'ts_inds'.")

    # Contract
    for x, y in track(map(sorted, path),
                      console=Console(stderr=True),
                      description="Contracting...",
                      total=len(path),
                      disable=(verbose <= 0)):
        if x == y:
            raise ValueError("'path' is not valid.")

        # Get indices
        ys = ts_inds.pop(y)
        xs = ts_inds.pop(x)

        # Get arrays
        if arrays is not None:
            ay = arrays.pop(y)
            ax = arrays.pop(x)

        # Get all indices
        all_inds = frozenset(xs) | frozenset(ys)

        # Get shared indices
        shared_inds = frozenset(xs) & frozenset(ys)

        # All shared inds must have a hyper-count > 0
        assert all(
            map(lambda x: x[0] not in shared_inds or x[1] > 0,
                hyper_count.items()))

        # Get hyper inds
        hyper_inds = frozenset(
            map(
                op.itemgetter(0),
                filter(lambda x: x[0] in shared_inds and x[1] > 1,
                       hyper_count.items())))

        # Update hyper-count
        for x in shared_inds:
            hyper_count[x] -= 1

        # Generate new inds
        zs = (frozenset(xs) ^ frozenset(ys)) | hyper_inds | (output_inds &
                                                             all_inds)

        # Sort new inds as they appear in xs and ys
        zs = tuple(
            mit.unique_everseen(
                its.chain(filter(lambda x: x in zs, xs),
                          filter(lambda y: y in zs, ys))))

        # Append new tensor
        if arrays is not None:
            arrays.append(
                do('einsum', tensor_utils.get_einsum_path(xs, ys, zs), ax, ay))

        # Append new indices
        ts_inds.append(zs)

    # Get the new output indices
    output_inds = output_inds.intersection(mit.flatten(ts_inds))

    # Return new arrays
    return (ts_inds, output_inds) if arrays is None else (ts_inds, output_inds,
                                                          arrays)
