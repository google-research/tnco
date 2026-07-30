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
"""Tensor network utilities."""

from bisect import bisect_left
from collections import Counter
from collections import defaultdict
import functools as fts
import itertools as its
import math
import operator as op
from random import Random
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple, Union
from warnings import warn

import autoray as ar
import more_itertools as mit
import opt_einsum as oe
from rich.console import Console
from rich.progress import Progress
from rich.progress import track

from tnco.ordered_frozenset import OrderedFrozenSet
from tnco.typing import Array
from tnco.typing import Index
from tnco.typing import TensorName
import tnco.utils.tensor as tensor_utils

__all__ = [
    'get_random_contraction_path', 'get_symbol', 'get_einsum_subscripts',
    'read_inds', 'fuse', 'decompose_hyper_inds', 'merge_contraction_paths',
    'split_contraction_path', 'contract', 'get_hyper_count'
]


def get_connected_components(ts_inds: Iterable[List[Index]],
                             verbose: int = 0) -> List[List[int]]:
    ts_inds_list = list(ts_inds)
    num_tensors = len(ts_inds_list)

    parent = list(range(num_tensors))
    rank = [0] * num_tensors

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            if rank[root_i] < rank[root_j]:
                parent[root_i] = root_j
            elif rank[root_i] > rank[root_j]:
                parent[root_j] = root_i
            else:
                parent[root_j] = root_i
                rank[root_i] += 1

    index_to_tensor = {}

    for tensor_id, inds in track(enumerate(ts_inds_list),
                                 description="Finding CC...",
                                 total=num_tensors,
                                 console=Console(stderr=True),
                                 disable=(verbose <= 0)):
        for index_name in inds:
            if index_name in index_to_tensor:
                union(tensor_id, index_to_tensor[index_name])
            else:
                index_to_tensor[index_name] = tensor_id

    components = {}
    for tensor_id in range(num_tensors):
        root = find(tensor_id)
        if root not in components:
            components[root] = []
        components[root].append(tensor_id)

    return list(map(tuple, map(sorted, components.values())))


def get_random_contraction_path(
        ts_inds: Iterable[List[Index]],
        output_inds: Optional[Iterable[Index]] = None,
        *,
        merge_paths: bool = True,
        autocomplete: bool = True,
        seed: Optional[int] = None,
        verbose: int = False,
        **kwargs) -> Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]]:
    """Generates a random contraction path.

    Generates a random contraction path for the given tensor indices.

    Args:
        ts_inds: List of indices for each tensor.
        output_inds: (Deprecated) List of output indices.
        merge_paths: If ``True``, merges all paths even if tensors are
            disconnected. If ``False``, returns separate paths for each
            connected component.
        autocomplete: If ``True`` and ``merge_paths=True``, connects
            disconnected components.
        seed: Random seed.
        verbose: If ``True``, prints verbose output.

    Returns:
        A list of contraction steps (linear einsum format). If
        ``merge_paths=False``, returns a list of paths for each connected
        component. It is guaranteed that only tensors that share at least one
        index are contracted in connected paths.

    Notes:
        Hyper-indices that are also output indices are removed from
        ``output_inds`` before contraction to guarantee that only tensors
        that share an index are contracted. In ``opt_einsum``'s greedy
        algorithm, indices listed in ``output_inds`` are ignored as
        structural contraction edges during its primary inner-product phase
        (Phase 2). If a connecting hyper-index is left in ``output_inds``,
        ``opt_einsum`` treats the connected tensors as fragmented and may
        combine them via arbitrary size-based outer products in Phase 3.
        Removing hyper-indices from ``output_inds`` ensures ``opt_einsum``
        treats them as summed-over internal edges, guaranteeing that only
        tensors sharing at least one index are paired.

    Examples:
        >>> from tnco.utils.tn import get_random_contraction_path
        >>> ts_inds = [['i', 'j'], ['j', 'k'], ['k', 'l']]
        >>> get_random_contraction_path(ts_inds, seed=42)
        [(0, 1), (0, 1)]
    """
    if output_inds is not None:
        warn(
            "'output_inds' is deprecated, "
            "and it will be removed in version '0.2'.",
            DeprecationWarning,
            stacklevel=2)

    _return_contraction = kwargs.pop('_return_contraction', False)
    if kwargs:
        raise TypeError("Got an expected keyword argument(s).")

    rng = Random(seed)
    ts_inds = list(ts_inds)
    n_tensors = len(ts_inds)

    if output_inds is None:
        base_hyper_count = get_hyper_count(ts_inds)
        output_inds_set = OrderedFrozenSet(
            x for x, count in base_hyper_count.items() if count == 0)
    else:
        output_inds_set = OrderedFrozenSet(output_inds)

    hyper_count = get_hyper_count(ts_inds, output_inds=output_inds_set)
    filtered_output_inds = OrderedFrozenSet(
        x for x in output_inds_set if hyper_count.get(x, 0) <= 1)

    avail_inds_cc = get_connected_components(ts_inds, verbose=verbose)

    paths = []
    next_id = n_tensors

    for i, cc in track(
            enumerate(avail_inds_cc),
            description="Getting contraction paths ...",
            total=len(avail_inds_cc),
            disable=(verbose <= 0),
            console=Console(stderr=True),
    ):
        if len(cc) <= 1:
            paths.append([])
            continue

        cc_list = list(cc)
        rng.shuffle(cc_list)

        ts_inds_cc = [ts_inds[idx] for idx in cc_list]
        output_inds_cc = filtered_output_inds.intersection(
            mit.flatten(ts_inds_cc))

        subscripts = get_einsum_subscripts(ts_inds_cc, output_inds_cc)
        shapes = [(2,) * len(xs) for xs in ts_inds_cc]
        linear_path_cc, _ = oe.contract_path(subscripts,
                                             *shapes,
                                             shapes=True,
                                             optimize='greedy')

        loc = list(cc_list)
        path_cc = []
        for px, py in linear_path_cc:
            px, py = sorted((px, py))
            ty = loc.pop(py)
            tx = loc.pop(px)
            tz = next_id
            next_id += 1
            loc.append(tz)
            path_cc.append((tx, ty, tz))
        paths.append(path_cc)

    if _return_contraction:
        return paths

    linear_paths = []
    for i, path in enumerate(paths):
        linear_path = []
        loc = list(range(n_tensors))
        for x, y, z in track(
                path,
                description=
                f"Mapping to original tensor order ({i + 1}/{len(paths)}) ...",
                disable=(verbose <= 1),
                console=Console(stderr=True),
        ):
            px, py = sorted(map(lambda x: bisect_left(loc, x), (x, y)))
            loc.pop(py)
            loc.pop(px)
            loc.append(z)
            linear_path.append((px, py))
        linear_paths.append(linear_path)

    return (merge_contraction_paths(
        n_tensors,
        linear_paths,
        autocomplete=autocomplete,
        verbose=verbose - 1,
    ) if merge_paths else linear_paths)


def get_symbol(i: int) -> str:
    """Returns a unique symbol for a given integer.

    Maps an integer to a character for einsum notation.

    Args:
        i: Integer index.

    Returns:
        A unique character symbol.
    """
    # standard a-z, A-Z
    if i < 52:
        return "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]

    # Skip Unicode surrogates (0xD800-0xDFFF) 55296 is the start of the
    # surrogate range (0xD800). Add 2048 to skip past the entire surrogate
    # block (2048 chars)
    elif i >= 55296:
        return chr(i + 2048)

    # Start at 192 (0xC0)
    # 52 + 140 = 192
    else:
        return chr(i + 140)


def get_einsum_subscripts(ts_inds: Iterable[List[Index]],
                          output_inds: Optional[Iterable[Index]] = ()):
    """Generates einsum subscripts.

    Generates the einsum subscripts string for contracting multiple tensors.

    Args:
        ts_inds: List of indices for each tensor.
        output_inds: List of output indices.

    Returns:
        The einsum subscripts string.
    """
    # Convert to list
    ts_inds = list(ts_inds)
    output_inds = list(output_inds)

    # Map indices
    inds_map = dict(
        its.starmap(
            lambda i, x: (x, get_symbol(i)),
            enumerate(
                mit.unique_everseen(its.chain(mit.flatten(ts_inds),
                                              output_inds)))))

    # Return map
    return ','.join(map(lambda xs: ''.join(map(inds_map.get, xs)),
                        ts_inds)) + '->' + ''.join(
                            map(inds_map.get, output_inds))


def merge_contraction_paths(n_tensors: int,
                            paths: Iterable[List[Tuple[int, int]]],
                            *,
                            autocomplete: bool = True,
                            verbose: int = False) -> List[Tuple[int, int]]:
    """Merges contraction paths.

    Merges multiple contraction paths into a single path.

    Args:
        n_tensors: Total number of tensors.
        paths: List of contraction paths (each in linear format).
        autocomplete: If ``True``, adds steps to connect disconnected
            components.
        verbose: If ``True``, prints verbose output.

    Returns:
        A single merged contraction path in linear (einsum) format.

    Raises:
        ValueError: If paths are invalid.

    Examples:
        >>> from tnco.utils.tn import merge_contraction_paths
        >>> paths = [[(0, 1)], [(2, 3)]]
        >>> merge_contraction_paths(4, paths)
        [(0, 1), (0, 1), (0, 1)]
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


def split_contraction_path(
    n_tensors: int,
    path: Iterable[Tuple[int, int]],
    return_connected_components: bool = False,
    normalize_paths: bool = False,
    verbose: int = False
) -> Union[List[List[Tuple[int, int]]], Tuple[List[List[Tuple[int, int]]],
                                              List[FrozenSet[int]]]]:
    """Splits a contraction path.

    Splits a contraction path into disconnected components.

    Args:
        n_tensors: Total number of tensors.
        path: Contraction path in linear format.
        return_connected_components: If ``True``, returns the sets of tensors
            in each component.
        normalize_paths: If ``True``, re-indexes tensors in each sub-path.
        verbose: If ``True``, prints verbose output.

    Returns:
        A list of disconnected contraction paths in linear (einsum) format. If
        ``return_connected_components`` is ``True``, returns a tuple of paths
        and tensor sets for each connected component.

    Examples:
        >>> from tnco.utils.tn import split_contraction_path
        >>> path = [(0, 1), (0, 1)]
        >>> split_contraction_path(4, path)
        [[(0, 1)], [(2, 3)]]
    """

    # Convert path to list
    path = list(path)

    # Initialize tensors
    tensors = list(range(n_tensors))
    connectivity = [[] for _ in range(n_tensors + len(path) + 1)]

    # For each contraction ...
    n_intermediate_tensors = n_tensors
    for i, (x, y) in enumerate(
            track(map(sorted, path),
                  total=len(path),
                  description="Getting connected components...",
                  disable=verbose <= 0,
                  console=Console(stderr=True))):
        # Increase number of intermediate tensors
        n_intermediate_tensors += 1

        t_y = tensors.pop(y)
        t_x = tensors.pop(x)

        connectivity[t_x].append(i)
        connectivity[t_y].append(i)
        connectivity[n_intermediate_tensors].append(i)

        # Update list of tensors
        tensors.append(n_intermediate_tensors)

    # Split all the intermediate tensors in connected components
    cc = [
        c for c in get_connected_components(connectivity)
        if list(c) != [n_tensors]
    ]

    # Let's create the list of tensors and disconnected paths
    tensors = list(range(n_tensors))
    cc_tensors = list(map(sorted, cc)) if normalize_paths else list(
        list(range(n_tensors)) for _ in range(len(cc)))
    paths = list(mit.repeatfunc(list, len(cc)))

    # Let's start again
    n_intermediate_tensors = n_tensors
    for x, y in track(map(sorted, path),
                      total=len(path),
                      description="Getting disconnected paths...",
                      disable=verbose <= 0,
                      console=Console(stderr=True)):
        # Increase number of intermediate tensors
        n_intermediate_tensors += 1

        # Get tensors
        t_x, t_y = tensors[x], tensors[y]

        # Find the connected component
        cc_loc = mit.first(mit.locate(cc, lambda s: t_x in s))
        assert t_y in cc[cc_loc]

        # Update list of tensors
        tensors.pop(y)
        tensors.pop(x)
        tensors.append(n_intermediate_tensors)

        # Find the location of the intermediate tensors in the right connected
        # component
        x, y = sorted(map(cc_tensors[cc_loc].index, (t_x, t_y)))

        # Update path for the connected component
        paths[cc_loc].append(tuple((x, y)))

        # Update list of tensors for the connected component
        cc_tensors[cc_loc].pop(y)
        cc_tensors[cc_loc].pop(x)
        cc_tensors[cc_loc].append(n_intermediate_tensors)

    # Exclude the intermediate tensors
    if return_connected_components:
        cc = list(
            frozenset(filter(fts.partial(op.gt, n_tensors), s)) for s in cc)
        return paths, cc

    # Otherwise, just return the non-empty paths
    return list(filter(len, paths))


def read_inds(
    inds_map: Dict[Index, Tuple[int, TensorName]],
    *,
    output_index_token: TensorName = '*',
    sparse_index_token: TensorName = '/'
) -> Tuple[Dict[TensorName, Tuple[Index, ...]], Dict[Index, int],
           FrozenSet[Index], FrozenSet[Index]]:
    """Reads indices from a map.

    Constructs a tensor map from a dictionary of index information.

    Args:
        inds_map: Dictionary mapping indices to (dimension, tensor_names...).
        output_index_token: Token identifying output indices.
        sparse_index_token: Token identifying sparse indices.

    Returns:
        A tuple containing:
            - **tensor_map**: Map of tensor names to their indices.
            - **dims**: Map of index dimensions.
            - **output_inds**: Set of output indices.
            - **sparse_inds**: Set of sparse indices.

    Raises:
        ValueError: If tokens are identical.
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


def get_hyper_count(
        ts_inds: Iterable[Iterable[Index]],
        output_inds: Optional[Iterable[Index]] = None) -> Dict[Index, int]:
    """Computes the hyper-count for each index in a tensor network.

    The hyper-count of an index is defined as the number of times it is
    contracted (the number of tensors it appears in minus one), plus one
    if it is also an output index.

    Args:
        ts_inds: List of indices for each tensor.
        output_inds: Optional collection of output indices. If provided, the
            count for each output index is incremented by 1.

    Returns:
        A dictionary mapping each index to its hyper-count.
    """
    hyper_count = dict(
        its.starmap(lambda x, n: (x, n - 1),
                    Counter(mit.flatten(ts_inds)).items()))
    if output_inds is not None:
        for x_ in output_inds:
            hyper_count[x_] = hyper_count.get(x_, 0) + 1
    return hyper_count


def fuse(
    ts_inds: Iterable[List[Index]],
    dims: Union[int, Dict[Index, int]],
    max_width: float,
    output_inds: Optional[Iterable[Index]] = None,
    *,
    exclude_inds: Optional[Iterable[Index]] = (),
    seed: Optional[int] = None,
    return_fused_inds: bool = False,
    verbose: int = False
) -> Tuple[List[Tuple[int, int]], Optional[List[Tuple[Index]]]]:
    """Fuses tensors.

    Contracts tensors such that the resulting tensor width does not exceed
    ``max_width``.

    Args:
        ts_inds: List of indices for each tensor.
        dims: Dimensions of indices.
        max_width: Maximum allowed width. The width of a tensor is defined as
            the sum of the logarithms (base 2) of its dimensions.
        output_inds: Output indices.
        exclude_inds: Indices to exclude from contraction.
        seed: Random seed.
        return_fused_inds: If ``True``, returns indices of the fused tensors.
        verbose: If ``True``, prints verbose output.

    Returns:
        The contraction path. If ``return_fused_inds`` is ``True``, returns a
        tuple of (path, fused_indices).

    Raises:
        ValueError: If arguments are inconsistent.

    Examples:
        >>> from tnco.utils.tn import fuse
        >>> ts_inds = [['i', 'j'], ['j', 'k'], ['k', 'l']]
        >>> dims = {'i': 2, 'j': 2, 'k': 2, 'l': 2}
        >>> # Fuse tensors to limit width to 2 (log2(4))
        >>> fuse(ts_inds, dims, max_width=2, seed=42)
        [(0, 1), (0, 1)]
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
    hyper_count = get_hyper_count(ts_inds.values())

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
    """Decomposes hyper-indices in a tensor network.

    Decomposes diagonal tensors into hyper-indices.

    Args:
        arrays: List of tensor arrays.
        ts_inds: List of indices for each tensor.
        atol: Absolute tolerance for checking diagonality.

    Returns:
        A tuple containing:
            - **arrays**: Decomposed arrays.
            - **ts_inds**: New indices for decomposed tensors.
            - **hyper_inds_map**: Mapping from original to new indices.
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
    arrays: Optional[Iterable[Array]] = None,
    dims: Optional[Union[int, Dict[Index, int]]] = None,
    *,
    backend: Optional[str] = None,
    verbose: int = False
) -> Tuple[List[List[Index]], FrozenSet[Index], Optional[List[Array]]]:
    """Contracts a tensor network.

    Contracts a tensor network following a given path.

    Args:
        path: Contraction path in linear format.
        ts_inds: List of indices for each tensor.
        output_inds: Output indices.
        arrays: List of tensor arrays (optional).
        dims: Dimensions of indices (required if ``arrays`` is ``None``).
        backend: Backend for array operations (see ``autoray``).
        verbose: If ``True``, prints verbose output.

    Returns:
        A tuple containing:
            - **ts_inds**: Indices of remaining tensors (should be one).
            - **output_inds**: Final output indices.
            - **arrays**: (Optional) Resulting array(s) if ``arrays`` was
              provided.

    Raises:
        ValueError: If arguments are inconsistent or path is invalid.

    Examples:
        >>> import numpy as np
        >>> from tnco.utils.tn import contract
        >>> path = [(0, 1)]
        >>> ts_inds = [['i', 'j'], ['j', 'k']]
        >>> arrays = [np.eye(2), np.ones((2, 2))]
        >>> inds, output, res = contract(path, ts_inds, arrays=arrays)
        >>> inds
        [('i', 'k')]
        >>> res[0]
        array([[1., 1.],
               [1., 1.]])
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
    hyper_count = get_hyper_count(ts_inds)

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
                       hyper_count.items()))) | (output_inds & shared_inds)

        # Update hyper-count
        for x in shared_inds:
            hyper_count[x] -= 1

        # Contract the tensors
        if arrays is None:
            zs = tensor_utils.tensordot((None, xs), (None, ys),
                                        hyper_inds=hyper_inds,
                                        return_inds_only=True)

        else:
            (az, zs) = tensor_utils.tensordot((ax, xs), (ay, ys),
                                              hyper_inds=hyper_inds)
            arrays.append(az)

        # Append new indices
        ts_inds.append(zs)

    # Get the new output indices
    output_inds = output_inds.intersection(mit.flatten(ts_inds))

    # Return new arrays
    return (ts_inds, output_inds) if arrays is None else (ts_inds, output_inds,
                                                          arrays)
