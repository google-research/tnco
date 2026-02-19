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
"""Cost model for infinite memory optimization."""

from importlib import import_module
from typing import Dict, FrozenSet, Iterable, Literal, Optional, Union

import more_itertools as mit

from tnco.bitset import Bitset
from tnco.typing import Index

__all__ = ['SimpleCostModel']


class BaseCostModel:
    """Base class for cost models.

    Base class for cost models.
    """

    def contraction_cost(*args, **kwargs):
        raise NotImplementedError()


class SimpleCostModel(BaseCostModel):
    """Simple cost model.

    Return the contraction cost between two tensors. The contraction cost in
    this simple model corresponds to:

    cost = prod_i d_i

    where i are all the indices between the two tensors (counted once). The
    model also takes into account the presence of sparse indices. In this case,
    the cost of the contraction corresponds to:

    cost = prod_j d_j * min(prod_k d_k, n_projs)

    where j are all the non-sparse indices between the two tensors (counted
    once), k are all the sparse-indices between the two tensors (counted once),
    and n_projs are the maximum number of allowed projections among all the
    sparse indices. While this model does not account for the actual
    configurations of sparse indices, it is a good proxy for the actual cost
    when the actual configurations of sparse indices are random enough.

    Args:
        cost_type: The type to use to represent the cost of contraction while
            optimizing the tensor network contraction.
        sparse_inds: List of indices that are sparse.
        n_projs: Total number of configurations for the sparse indices.
    """

    def __init__(self,
                 *,
                 cost_type: Literal['float32', 'float64', 'float128',
                                    'float1024'] = 'float64',
                 sparse_inds: Optional[Iterable[Index]] = None,
                 n_projs: Optional[int] = None):

        # Check type
        cost_type = str(cost_type).lower()
        if cost_type not in ['float32', 'float64', 'float128', 'float1024']:
            raise NotImplementedError(
                f"'cost_type={self.cost_type}' not available.")

        # Check sparse_inds and n_projs
        if n_projs is not None and (n_projs != int(n_projs) or n_projs < 1):
            raise ValueError("'n_projs' must be a positive number.")
        sparse_inds = None if sparse_inds is None else frozenset(sparse_inds)
        if sparse_inds is None and n_projs:
            raise ValueError("'n_projs' cannot be specified if "
                             "'sparse_inds' is not provided.")
        if sparse_inds and not n_projs:
            raise ValueError(
                "'n_projs' must be specified if 'sparse_inds' is provided.")

        # Save args
        self._cost_type = cost_type
        self._sparse_inds = sparse_inds
        self._n_projs = n_projs
        self._mod = 'tnco_core.optimize.infinite_memory.cost_model'

    @property
    def cost_type(self) -> str:
        """Cost type.

        Return the cost type to use while optimizing the tensor network
        contraction.

        Returns:
            The type to use for the cost.
        """
        return self._cost_type

    @property
    def sparse_inds(self) -> FrozenSet[Index]:
        """Sparse indices.

        Return a set of the sparse indices.

        Returns:
            A set of sparse indices.
        """
        return self._sparse_inds

    @property
    def n_projs(self) -> int:
        """Total number of projections.

        The total number of projections among all the sparse indices.

        Returns:
            The total number of projections among all the sparse indices.
        """
        return self._n_projs

    def contraction_cost(self, inds_A: Iterable[Index], inds_B: Iterable[Index],
                         inds_C: Iterable[Index], dims: Union[Dict[Index, int],
                                                              int]) -> float:
        """Contraction cost.

        Return the cost of contracting 'inds_A' with 'inds_B', to return
        'inds_C'. It also takes into account any sparse index and the presence
        of sliced indices.

        Args:
            inds_A: The indices of tensor A.
            inds_B: The indices of tensor B.
            inds_C: The indices of the tensor after contracting A with B.
            dims: The dimensions for each index.

        Returns:
            The contraction cost.

        Raises:
            ValueError: The arguments are not consistent with each other.
        """

        # Convert to tuple
        inds_A = tuple(inds_A)
        inds_B = tuple(inds_B)
        inds_C = tuple(inds_C)

        # Get all inds
        all_inds = tuple(mit.unique_everseen(inds_A + inds_B + inds_C))
        if self.sparse_inds is not None:
            all_inds = tuple(
                mit.unique_everseen(all_inds + tuple(self.sparse_inds)))

        # Try to convert dims to int
        try:
            dims = int(dims)
        except (ValueError, TypeError):
            pass

        # Check dimensions, and convert it to tuple for the given order
        if not isinstance(dims, int):
            if not frozenset(dims).issuperset(all_inds):
                raise ValueError("Some dimensions are missing.")
            dims = tuple(map(dims.get, all_inds))

        # Get core
        core = self.__get_core__(all_inds)

        # Convert
        inds_A, inds_B, inds_C = map(
            lambda xs: Bitset(map(all_inds.index, xs), len(all_inds)),
            (inds_A, inds_B, inds_C))

        # Get result
        return core.contraction_cost(inds_A, inds_B, inds_C, dims)

    def __get_core__(self, inds_order: Iterable[Index]):
        # Convert
        inds_order = tuple(inds_order)

        # Get right core and arguments
        if self.sparse_inds:
            # Check
            if not self.sparse_inds.issubset(inds_order):
                raise ValueError(
                    "Sparse indices are not a subset of 'inds_order'.")

            core_name = 'SimpleCostModelSparseInds'
            sparse_inds = Bitset(map(inds_order.index, self.sparse_inds),
                                 len(inds_order))
            args = (sparse_inds, self.n_projs)

        else:
            core_name = 'SimpleCostModel'
            args = ()

        # Return core
        return getattr(import_module(self._mod),
                       f'{core_name}_{self.cost_type}')(*args)

    def __repr__(self):
        repr_ = 'SimpleCostModel('
        repr_ += f'cost_type={self.cost_type}'
        if self.sparse_inds is not None:
            repr_ += f', sparse_inds={self.sparse_inds}'
            repr_ += f', n_projs={self.n_projs}'
        repr_ += ')'
        return repr_

    def __eq__(self, other):
        return all(
            map(lambda x: getattr(self, x) == getattr(other, x),
                ('cost_type', 'sparse_inds', 'n_projs')))
