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
"""Contraction Tree."""

import functools as fts
import itertools as its
import math
import operator as op
from collections import Counter
from types import MappingProxyType
from typing import (Any, Callable, Dict, FrozenSet, Iterable, List, NoReturn,
                    Optional, Tuple, Union)

import more_itertools as mit
from rich.console import Console
from rich.progress import Progress, track
from tnco_core import ContractionTree as _ContractionTree
from tnco_core import Node
from tnco_core.utils import get_contraction, traverse

from tnco.typing import Index

__all__ = ['ContractionTree', 'traverse_tree']


class ContractionTree(_ContractionTree):
    """Contraction tree.

    Class representing a contraction tree.

    Args:
        path: Path used to create the contraction tree.
        ts_inds: List of indices for each tensor.
        dims: Dimensions of each index.
        output_inds: The list of output indices. It must be provided if
            ``ts_inds`` has hyper-indices.
        check_shared_inds: Check if all contracted tensors share at least one
            index.
        verbose: If ``True``, prints verbose output.

    Raises:
        ValueError: If the provided parameters are not consistent.
        ValueError: If ``check_shared_inds`` is ``True`` and the contraction
            path has contracted tensors that do not share an index.

    Examples:
        >>> from tnco.ctree import ContractionTree
        >>> path = [(0, 1)]
        >>> ts_inds = [['i', 'j'], ['j', 'k']]
        >>> dims = {'i': 2, 'j': 2, 'k': 2}
        >>> ctree = ContractionTree(path, ts_inds, dims)
        >>> ctree.max_width()
        2.0
    """

    def __init__(self,
                 path: Iterable[Tuple[int, int]],
                 ts_inds: Iterable[List[Index]],
                 dims: Union[Dict[Index, int], int],
                 *,
                 output_inds: Optional[Iterable[Index]] = None,
                 check_shared_inds: bool = False,
                 verbose: bool = False,
                 **kwargs) -> None:
        # Get cache if present
        _cache = kwargs.pop('_cache', None)
        if kwargs:
            raise TypeError("Got unexpected keyword arguments.")

        # Convert to list
        ts_inds = list(ts_inds)

        # If a list of nodes has been provide, use it as 'ctree'
        if all(map(lambda x: isinstance(x, Node), path)):
            ctree = path

            if output_inds is not None:
                raise ValueError(
                    "'output_inds' cannot be provided if a contraction "
                    "tree is used instead of a path.")

            # Get cached positions of relevant tensors
            self._n_tensors = int(_cache[0])
            self._tensors_pos = tuple(_cache[1])

            # Get cached order of inds
            self._inds_order = tuple(_cache[2])
            if frozenset(self._inds_order) != frozenset(mit.flatten(ts_inds)):
                raise ValueError("'_inds_order' is not valid.")

        # Otherwise, build 'ctree' from path
        else:
            # Get number of tensors
            n_tensors = len(ts_inds)

            # Initialize contraction
            contraction = []

            # Get contraction
            pos_ = list(range(len(ts_inds)))
            for i_, xs_ in enumerate(path):
                x_, y_ = sorted(xs_)
                py_ = pos_.pop(y_)
                px_ = pos_.pop(x_)
                pos_.append(i_ + len(ts_inds))
                contraction.append((px_, py_, pos_[-1]))

            # Get only relevant positions of tensors
            self._n_tensors = n_tensors
            self._tensors_pos = tuple(
                sorted(
                    filter(lambda x: x < len(ts_inds),
                           mit.unique_everseen(mit.flatten(contraction)))))

            # Get all inds
            all_inds = tuple(
                mit.unique_everseen(
                    mit.flatten(map(lambda x: ts_inds[x], self._tensors_pos))))

            # Get hyper-count
            hyper_count = dict(
                its.starmap(
                    lambda x, n: (x, n - 1),
                    Counter(
                        mit.flatten(map(lambda x: ts_inds[x],
                                        self._tensors_pos))).items()))

            # Get output inds
            if output_inds is None:
                if any(map(lambda x: x > 1, hyper_count.values())):
                    raise ValueError(
                        "'output_inds' must be provided if 'ts_inds' "
                        "has hyper-indices.")
                output_inds = frozenset(
                    map(op.itemgetter(0),
                        filter(lambda x: x[1] == 0, hyper_count.items())))
            else:
                output_inds = frozenset(output_inds)

            # Ignore extra output inds
            output_inds = output_inds.intersection(all_inds)

            # Increment hyper-count for hyper-inds
            for x_ in output_inds:
                hyper_count[x_] += 1

            # Build tensors
            ts_inds.extend([None] *
                           (max(mit.flatten(contraction)) - n_tensors + 1))

            for tx_, ty_, tz_ in track(
                    contraction,
                    console=Console(stderr=True),
                    disable=(verbose <= 0),
                    total=len(contraction),
                    description="Creating intermediate tensors..."):
                # Get inds
                ix_ = frozenset(ts_inds[tx_])
                iy_ = frozenset(ts_inds[ty_])

                # Get shared
                shared_ = ix_ & iy_
                if check_shared_inds and not shared_:
                    raise ValueError("'check_shared_inds' failed.")

                # Get new inds
                iz_ = ix_ ^ iy_

                # Update hyper count
                for is_ in shared_:
                    assert hyper_count[is_] > 0
                    hyper_count[is_] -= 1
                    if hyper_count[is_] > 0:
                        iz_ |= {is_}

                # Append new inds to tensors
                ts_inds[tz_] = tuple(iz_)

            # Get all positions
            pos_ = sorted(mit.unique_everseen(mit.flatten(contraction)))

            # Check positions with cached ones
            assert len(pos_) >= len(self._tensors_pos) and tuple(
                pos_[:len(self._tensors_pos)]) == self._tensors_pos

            # Get tree map
            tree_map_ = dict(zip(pos_, range(len(pos_))))
            tree_ = list(
                map(lambda xs: tuple(map(tree_map_.get, xs)), contraction))

            # Get tree nodes
            ctree = [[-1, -1, -1] for _ in range(max(mit.flatten(tree_)) + 1)]
            for x_, y_, z_ in tree_:
                ctree[x_][2] = z_
                ctree[y_][2] = z_
                ctree[z_][:2] = [x_, y_]

            # Get inds
            ts_inds = list(map(lambda x: ts_inds[x], pos_))

            # Update dimensions
            try:
                dims = dict(
                    map(lambda x: (x, dims[x]),
                        mit.unique_everseen(mit.flatten(ts_inds))))
            except TypeError:
                if int(dims) != dims:
                    raise ValueError("'dims' is not valid.")
                dims = dict(
                    zip(mit.unique_everseen(mit.flatten(ts_inds)),
                        its.repeat(int(dims))))

            # Get order of inds
            self._inds_order = tuple(mit.unique_everseen(mit.flatten(ts_inds)))

        def get_node(node):
            # If Node, just return
            if isinstance(node, Node):
                return node

            # Otherwise, convert to Node
            x, y, z = map(lambda x: -1 if x is None else x, node)
            return Node((x, y), z)

        # Get inds map
        inds_map_ = dict(zip(self._inds_order, range(len(self._inds_order))))

        # Convert
        nodes = tuple(map(get_node, ctree))
        ts_inds = tuple(map(lambda xs: tuple(map(inds_map_.get, xs)), ts_inds))
        try:
            dims = (int(dims),) * len(self._inds_order)
        except (ValueError, TypeError):
            dims = tuple(map(dims.get, self._inds_order))

        # Build core contraction tree
        super().__init__(nodes,
                         ts_inds,
                         dims,
                         check_shared_inds=check_shared_inds)

    @staticmethod
    def __build__(*args) -> 'ContractionTree':
        nodes, ts_inds, dims, _cache = args
        return ContractionTree(nodes, ts_inds, dims, _cache=_cache)

    def __reduce__(self) -> Tuple[Any, ...]:
        return self.__build__, (self.nodes, self.inds[:], dict(self.dims),
                                (self._n_tensors, self._tensors_pos,
                                 self._inds_order))

    def __eq__(self, other: Any) -> bool:
        return super().__eq__(other) and self._inds_order == other._inds_order

    def __repr__(self) -> str:
        return f'ContractionTree(n_nodes={len(self)}, n_inds={self.n_inds})'

    def all_inds(self) -> FrozenSet[Index]:
        """Returns all indices.

        Returns all indices included in this contraction tree.

        Returns:
            All indices included in this contraction tree.
        """
        return frozenset(self._inds_order)

    def output_inds(self) -> FrozenSet[Index]:
        """Returns output indices.

        Returns the output indices of the contraction tree.

        Returns:
            The output indices in this contraction tree.
        """
        return self.inds[-1]

    @property
    def nodes(self) -> List[Node]:
        """Returns nodes.

        Returns a list of nodes in the contraction tree.

        Returns:
            A list of nodes.
        """
        return super().nodes

    @property
    def inds(self) -> List[FrozenSet[Index]]:
        """Returns indices.

        Returns a list of indices for each node.

        Returns:
            A list of indices.
        """

        class IndsProxy:

            def __init__(self, inds, inds_map) -> None:
                self._inds_order = inds
                self._inds_map = inds_map

            def __getitem__(
                self, key: Union[int, slice]
            ) -> Union[FrozenSet[Index], Tuple[FrozenSet[Index], ...]]:

                def get_inds(xs):
                    return frozenset(
                        map(lambda x: self._inds_map[x], xs.positions()))

                if isinstance(key, int):
                    return get_inds(self._inds_order[key])

                return tuple(map(get_inds, self._inds_order[key]))

        # Return all inds
        return IndsProxy(super().inds, self._inds_order)

    @property
    def dims(self) -> Dict[Any, int]:
        """Map of dimensions.

        Returns a map of dimensions for each index.

        Returns:
            Dimensions for each index.
        """
        # Get dimensions from base
        dims = super().dims

        # Return a dict
        return MappingProxyType(
            dict(
                zip(self._inds_order,
                    its.repeat(dims) if isinstance(dims, int) else dims)))

    def path(self) -> List[Tuple[int, int]]:
        """Returns contraction path in linear (einsum) format.

        Returns the contraction path in linear (einsum) format.

        Returns:
            The contraction path.
        """
        # Get full contraction
        contraction = get_contraction(self)

        # Rescale contraction
        shift = self._n_tensors - self.n_leaves

        def rescale(pos):
            nonlocal shift
            return self._tensors_pos[pos] if pos < len(
                self._tensors_pos) else pos + shift

        contraction = list(map(lambda xs: tuple(map(rescale, xs)), contraction))

        # Get all positions
        all_pos = list(range(self._n_tensors))

        # Initialize path
        path = []

        # Convert to path
        for *xs_, z_ in contraction:
            pos_ = tuple(map(all_pos.index, xs_))
            path.append(pos_)
            if pos_[0] > pos_[1]:
                pos_ = pos_[1], pos_[0]
            all_pos.pop(pos_[1])
            all_pos.pop(pos_[0])
            all_pos.append(z_)

        # Return path in linear (einsum) format
        return path

    def max_width(self) -> float:
        """Maximum width.

        Calculates the maximum width among all tensors in the contraction tree.
        The width of a tensor is defined as the sum of the logarithms base 2 of
        the dimensions of its indices.

        Returns:
            The maximum width.
        """
        return max(
            map(
                math.log2,
                map(lambda xs: fts.reduce(op.mul, map(self.dims.get, xs), 1),
                    self.inds)))


def traverse_tree(ctree: ContractionTree,
                  callback: Callable[[int], NoReturn],
                  *,
                  verbose: int = False) -> NoReturn:
    """Traverses ``tree`` and calls ``callback`` for each node.

    Traverses ``tree`` and calls ``callback`` for each node of the tree. The
    ``callback`` must accept an integer as an argument, which corresponds to
    the position of the node in the contraction tree.

    Args:
        ctree: Contraction tree to traverse.
        callback: The callback function to call.
        verbose: If ``True``, prints verbose output.
    """
    with Progress(disable=(verbose <= 0), console=Console(stderr=True)) as pbar:
        task = pbar.add_task("Exploring contraction tree...", total=len(ctree))

        # Add pbar to callback
        def callback_(pos: int):
            pbar.update(task, advance=1)
            callback(pos)

        # Traverse ctree
        traverse(ctree, callback_)

        # Final update of pbar
        pbar.update(task, refresh=True)
