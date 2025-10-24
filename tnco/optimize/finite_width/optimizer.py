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

from importlib import import_module
from typing import (Any, FrozenSet, Iterable, Literal, NoReturn, Optional,
                    TypeVar, Union)

from tnco.bitset import Bitset
from tnco.ctree import ContractionTree
from tnco.typing import Index

from .cost_model import BaseCostModel

__all__ = ['Optimizer']

BaseProbability = TypeVar('BaseProbability')


class Optimizer:
    """Optimize contraction tree.

    Optimize contraction tree.

    Args:
        ctree: Contraction tree to optimize.
        cmodel: Cost model to use for the optimization.
        slice_update: How slices are updated.
        max_number_new_slices: Maximum number of slices to add if the
            contraction cannot fit in memory.
        skip_slices: Indices in 'skip_slices' will not be sliced.
        seed: Seed to use to initialize random number generator.
        disable_shared_inds: If 'False', it is guaranteed that two contracted
            tensors always share an index.
        atol: Check cache with given precision.

    Raises:
        NotImplementedError: If 'cmodel' or 'slice_update' are not supported.
        ValueError: If arguments are not consistent with each other.
    """

    def __init__(self,
                 ctree: ContractionTree,
                 cmodel: BaseCostModel,
                 *,
                 slice_update: Optional[Literal['greedy']] = 'greedy',
                 max_number_new_slices: int = 0,
                 skip_slices: Iterable[Index] = None,
                 seed: Optional[Union[int, str]] = None,
                 disable_shared_inds: Optional[bool] = False,
                 atol: Optional[float] = 1e-5,
                 **kwargs):

        # Check cost model
        if not isinstance(cmodel, BaseCostModel):
            raise NotImplementedError("f'{cmodel}' not supported.")

        # Get cache if present
        _min_ctree = kwargs.pop('_min_ctree', None)
        _slices = kwargs.pop('_slices', None)
        _min_slices = kwargs.pop('_min_slices', None)
        if kwargs:
            raise TypeError("Got unexpected keyword arguments.")

        # Check slice_update
        slice_update = str(slice_update).lower()
        if slice_update not in ['greedy']:
            raise NotImplementedError(f"'{slice_update=}' not available.")

        # Load module
        mod = import_module(f"tnco_core.optimize.finite_width.{slice_update}")

        # Load object
        obj = getattr(mod, f'Optimizer_{cmodel.cost_type}_{cmodel.width_type}')

        # Save model
        self._cmodel = cmodel

        # Save cache
        self._ctree_cache = (ctree.dims, ctree._n_tensors, ctree._tensors_pos,
                             ctree._inds_order)

        # Check slices that must be skipped
        self._skip_slices = frozenset((
        ) if skip_slices is None else skip_slices)
        if not self.skip_slices.issubset(ctree._inds_order):
            raise ValueError(
                "'skip_slices' must be a subset of available indices.")

        # Check that tensors can fit the max allowed width even if some inds
        # are skipped
        if any(
                map(
                    lambda xs: self.cmodel.width(xs & self.skip_slices, ctree.
                                                 dims) > self.cmodel.max_width,
                    ctree.inds)):
            raise ValueError("Too many indices in 'skip_slices'.")

        # Convert slices to bitset
        if _slices is not None:
            _slices = Bitset(map(ctree._inds_order.index, _slices),
                             ctree.n_inds)
        if _min_slices is not None:
            _min_slices = Bitset(map(ctree._inds_order.index, _min_slices),
                                 ctree.n_inds)

        # Get bitset corresponding to the slices to skip
        _skip_slices = Bitset(map(ctree._inds_order.index, self.skip_slices),
                              ctree.n_inds) if len(self.skip_slices) else None

        # Get C++ core
        self._optimizer = obj(ctree,
                              cmodel.__get_core__(ctree._inds_order),
                              max_number_new_slices=max_number_new_slices,
                              seed=seed,
                              disable_shared_inds=disable_shared_inds,
                              atol=atol,
                              skip_slices=_skip_slices,
                              min_ctree=_min_ctree,
                              slices=_slices,
                              min_slices=_min_slices)

    def __get_ctree__(self, ctree):
        return ContractionTree(
            path=ctree.nodes,
            ts_inds=map(
                lambda xs: tuple(
                    map(lambda x: self._ctree_cache[3][x], xs.positions())),
                ctree.inds),
            dims=self._ctree_cache[0],
            _cache=self._ctree_cache[1:])

    @property
    def ctree(self) -> ContractionTree:
        """Contraction Tree.

        Returns the contraction tree.

        Returns:
            The contraction tree.
        """
        return self.__get_ctree__(self._optimizer.ctree)

    @property
    def min_ctree(self) -> ContractionTree:
        """Contraction Tree with minimum contraction cost.

        Returns the contraction tree with the minimum contraction cost found
        during the optimization.

        Returns:
            The contraction tree.
        """
        return self.__get_ctree__(self._optimizer.min_ctree)

    @property
    def cmodel(self) -> BaseCostModel:
        """Cost model.

        Return the cost model used during the optimization.

        Returns:
            The cost model.
        """
        return self._cmodel

    @property
    def skip_slices(self) -> FrozenSet[Index]:
        """Indices to skip while slicing.

        Returns the set of indices which are not sliced to fit within the given
        maximum width.

        Returns:
            The set of indices to skip while slicing.
        """
        return self._skip_slices

    @property
    def disable_shared_inds(self) -> bool:
        """Flag to disable shared indices.

        Flag to disable shared indices. If 'disbale_shared_inds=False', it is
        guaranteed that every pair of contracted tensors always share at least
        an index.

        Returns:
            Flag to disable shared indices.
        """
        return self._optimizer.disable_shared_inds

    @property
    def prng_state(self) -> Any:
        """State of the pseudo random number generator.

        Returns the state of the pseudo random number generator used during the
        optimization.

        Returns:
            The state of the pseudo random number generator.
        """
        return self._optimizer.prng_state

    @property
    def total_cost(self) -> float:
        """Contraction cost of 'Optimizer.ctree'.

        Returns the contraction cost of 'Optimizer.ctree'.

        Returns:
            The contraction cost of 'Optimizer.ctree'.
        """
        return self._optimizer.total_cost

    @property
    def min_total_cost(self) -> float:
        """Contraction cost of 'Optimizer.min_ctree'.

        Returns the contraction cost of 'Optimizer.min_ctree'.

        Returns:
            The contraction cost of 'Optimizer.min_ctree'.
        """
        return self._optimizer.min_total_cost

    @property
    def log2_total_cost(self) -> float:
        """Log2 of the contraction cost of 'Optimizer.ctree'.

        Returns the logarithm in base 2 of the contraction cost of
        'Optimizer.ctree'.

        Returns:
            The logarithm in base 2 of the contraction cost of
            'Optimizer.ctree'.
        """
        return self._optimizer.log2_total_cost

    @property
    def log2_min_total_cost(self) -> float:
        """Log2 of the contraction cost of 'Optimizer.min_ctree'.

        Returns the logarithm in base 2 of the contraction cost of
        'Optimizer.min_ctree'.

        Returns:
            The logarithm in base 2 of the contraction cost of
            'Optimize.min_ctree'.
        """
        return self._optimizer.log2_min_total_cost

    @property
    def slices(self) -> FrozenSet[Index]:
        """Slices used by 'Optimizer.ctree'.

        Returns the sliced indices used in 'Optimizer.ctree' to keep every
        tensor in the contraction tree within the allowed maximum width.

        Returns:
            The sliced indices used in 'Optimizer.ctree'.
        """
        return frozenset(
            map(lambda x: self._ctree_cache[-1][x],
                self._optimizer.slices.positions()))

    @property
    def min_slices(self) -> FrozenSet[Index]:
        """Slices used by 'Optimizer.min_ctree'.

        Returns the sliced indices used in 'Optimizer.min_ctree' to keep every
        tensor in the contraction tree within the allowed maximum width.

        Returns:
            The sliced indices used in 'Optimizer.min_ctree'.
        """
        return frozenset(
            map(lambda x: self._ctree_cache[-1][x],
                self._optimizer.min_slices.positions()))

    def is_valid(self,
                 *,
                 atol: float = 1e-5,
                 return_message: str = False) -> bool:
        """Check if 'Optimizer' is in a valid state.

        Check if 'Optimizer' is in a valid state.

        Args:
            atol: Precision to use to check consistency.
            return_message: If 'return_message=True', it also returns the
                reason why 'Optimizer' is not in a valid state.

        Returns:
            If 'return_message=False', it returns a 'True' if 'Optimizer' is in
            a valid state. If 'return_message=True', it returns a tuple of a
            bool and a string, with the string being the reason why 'Optimizer'
            is not in a valid state.
        """
        return self._optimizer.is_valid(atol=atol,
                                        return_message=return_message)

    def update(self,
               prob: BaseProbability,
               *,
               update_slices: bool = True) -> NoReturn:
        """Update 'Optimizer' using 'prob'.

        When 'Optimizer.update' is called, a potential move in the contraction
        tree is proposed. 'prob' must take the difference in cost and the
        original cost, and decide to either accept or reject the move.

        Args:
            prob: The probability to use to either accept or reject the
                proposed move.
            update_slices: If 'True', also update the slices using the
                'Optimizer.slice_update' method.
        """
        self._optimizer.update(prob, update_slices=update_slices)

    @staticmethod
    def __build__(*args):
        (ctree, cmodel, prng_state, disable_shared_inds, min_ctree, slices,
         min_slices, skip_slices) = args
        return Optimizer(ctree,
                         cmodel,
                         seed=prng_state,
                         disable_shared_inds=disable_shared_inds,
                         skip_slices=skip_slices,
                         _min_ctree=min_ctree,
                         _slices=slices,
                         _min_slices=min_slices)

    def __reduce__(self):
        return self.__build__, (self.ctree, self.cmodel, self.prng_state,
                                self.disable_shared_inds, self.min_ctree,
                                self.slices, self.min_slices, self.skip_slices)

    def __repr__(self):
        return "Optimizer(ctree={}, cmodel={})".format(self.ctree, self.cmodel)

    def __eq__(self, other):
        return self.__reduce__()[1] == other.__reduce__()[1]
