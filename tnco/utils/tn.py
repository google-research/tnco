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
import math
import operator as op
from bisect import bisect_left
from collections import Counter, defaultdict
from random import Random
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple, Union
from warnings import warn

import autoray as ar
import more_itertools as mit
from rich.console import Console
from rich.progress import Progress, track

import tnco.utils.tensor as tensor_utils
from tnco.ordered_frozenset import OrderedFrozenSet
from tnco.typing import Array, Index, TensorName

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
            hyper-indices. (Deprecated in version '0.2')
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
        If 'merge_path=True', return a single path in linear (einsum)
        format. Otherwise, return multuple paths in linear (einsum)
        format for each connected component. Each path guarantees that
        only tensors that share at least one index are included in the
        path.
    """
    if output_inds is not None:
        warn(
            "'output_inds' is deprecated, "
            "and it will be removed in version '0.2'.",
            DeprecationWarning,
            stacklevel=2)

    # Extra args
    _return_contraction = kwargs.pop('_return_contraction', False)
    if kwargs:
        raise TypeError("Got an expected keyword argument(s).")

    # Initialize random number generator
    rng = Random(seed)

    # Convert to list
    ts_inds = list(ts_inds)

    # Store the initial number of tensors
    n_tensors = len(ts_inds)

    # First, let's map all indices to ints
    inds_map = dict(zip(mit.unique_everseen(mit.flatten(ts_inds)), its.count()))

    # Remap inds
    ts_inds = list(map(lambda xs: frozenset(map(inds_map.get, xs)), ts_inds))

    # Get map index->tensors
    index2tensors = mit.map_reduce(
        mit.flatten(
            its.starmap(lambda t, xs: zip(its.repeat(t), xs),
                        enumerate(ts_inds))), op.itemgetter(1),
        op.itemgetter(0))

    # Count how many times an index is contracted
    hyper_count = dict(
        filter(
            op.itemgetter(1),
            zip(index2tensors,
                map(lambda xs: len(xs) - 1, index2tensors.values()))))

    # Split indices in connected components
    color2inds = dict(map(lambda x: (x, frozenset([x])), range(len(inds_map))))
    index2color = dict(enumerate(range(len(inds_map))))
    for xs in track(ts_inds,
                    description="Getting connected components ...",
                    disable=(verbose <= 0),
                    console=Console(stderr=True)):
        if len(xs):
            # Get all colors
            colors = frozenset(map(index2color.get, xs))

            # Get all inds
            xs = fts.reduce(op.or_, map(color2inds.get, colors))

            # Update colors
            color2inds[mit.first(colors)] = xs
            index2color.update(zip(xs, its.repeat(mit.first(colors))))

    # Initialize paths
    paths = []

    # Swap location of two elements in an array
    def swap(a, x, y):
        a[x], a[y] = a[y], a[x]

    # Split the available indices in connected components
    avail_inds_cc = mit.map_reduce(index2color.items(), op.itemgetter(1),
                                   op.itemgetter(0)).values()

    # For each connected components ...
    for i, avail_inds in enumerate(avail_inds_cc):
        # Initialize contracted tensors
        contracted_tensors = set()

        # Initialize path
        path = []

        with Progress(disable=(verbose <= 0),
                      console=Console(stderr=True)) as pbar:

            # Size of the progress bar
            total_pbar = len(avail_inds)

            # Add progress bar
            task = pbar.add_task("Getting contraction path ({}/{}) ...".format(
                i + 1, len(avail_inds_cc)),
                                 total=total_pbar)

            # While there are available indices
            while avail_inds:
                # Select a random index
                swap(avail_inds, rng.randrange(len(avail_inds)), -1)
                index = avail_inds[-1]

                # If index has been already fully contracted, skip
                if index not in hyper_count or len(index2tensors[index]) <= 1:
                    avail_inds.pop()
                    continue

                # Get two random tensors
                tx, ty = rng.sample(list(
                    filter(lambda t: t not in contracted_tensors,
                           index2tensors[index])),
                                    k=2)

                # Get indices
                xs, ys = ts_inds[tx], ts_inds[ty]

                # Get shared inds
                shared = xs & ys

                # They should always share an index
                assert len(shared)

                # Update hyper-count for each shared index
                for x in shared:
                    hyper_count[x] -= 1
                    if hyper_count[x] == 0:
                        del hyper_count[x]

                # Get new set of indices
                tz = len(ts_inds)
                zs = (xs ^ ys).union(filter(lambda x: x in hyper_count, shared))

                # Update inds
                for x in zs:
                    index2tensors[x].append(tz)

                # Update tensors
                contracted_tensors |= {tx, ty}
                ts_inds.append(zs)

                # Update path
                path.append((tx, ty, tz))

                # Update pbar
                pbar.update(task, completed=total_pbar - len(avail_inds))

            # Final update
            pbar.update(task, completed=total_pbar, refresh=True)

            # Append to all paths
            paths.append(path)

    # For testing only
    if _return_contraction:
        return paths

    # Normalize paths
    linear_paths = []
    for i, path in enumerate(paths):
        linear_path = []
        loc = list(range(n_tensors))
        for x, y, z in track(
                path,
                description="Convert to linear (einsum) path ({}/{}) ...".
                format(i + 1, len(paths)),
                disable=(verbose <= 1),
                console=Console(stderr=True)):
            px, py = sorted(map(lambda x: bisect_left(loc, x), (x, y)))
            loc.pop(py)
            loc.pop(px)
            loc.append(z)
            linear_path.append((px, py))
        linear_paths.append(linear_path)

    # Merge paths if needed
    return merge_contraction_paths(
        n_tensors, linear_paths, autocomplete=autocomplete, verbose=verbose -
        1) if merge_paths else linear_paths


def merge_contraction_paths(
        n_tensors: int,
        paths: Iterable[List[Tuple[int, int]]],
        *,
        autocomplete: Optional[bool] = True,
        verbose: Optional[int] = False) -> List[Tuple[int, int]]:
    """Merge contraction paths.

    Merge contraction paths for disconnected tensor networks.

    Args:
        n_tensors: Number of total tensors
        paths: Contraction paths to merge in linear (einsum) format.
        autocomplete: If 'True', the merged path will include the contraction
            of the disconnected tensors.
        verbose: Verbose output.

    Returns:
        Merged path in linear (einsum) format.

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
        for x, y in track(path,
                          description="Merging paths ({}/{}) ...".format(
                              i + 1, len(paths)),
                          disable=(verbose <= 0),
                          console=Console(stderr=True)):

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
        The contraction path in linear (einsum) format. If
        'return_fused_inds=True', also return the corresponding indices
        of the fused tensors.

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
        path: Path to follow for the contraction in linear (einsum)
            format.
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
