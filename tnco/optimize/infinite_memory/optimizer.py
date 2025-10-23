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
from typing import Any, NoReturn, Optional, TypeVar, Union

from tnco.ctree import ContractionTree

from .cost_model import BaseCostModel

__all__ = ['Optimizer']

BaseProbability = TypeVar('BaseProbability')


class Optimizer:
    """
    Optimize contraction tree.

    Args:
        ctree: Contraction tree to optimize.
        cmodel: Cost model to use for the optimization.
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
                 seed: Optional[Union[int, str]] = None,
                 disable_shared_inds: Optional[bool] = False,
                 atol: Optional[float] = 1e-5,
                 **kwargs):

        # Check cost model
        if not isinstance(cmodel, BaseCostModel):
            raise NotImplementedError("f'{cmodel}' not supported.")

        # Get cache if present
        _min_ctree = kwargs.pop('_min_ctree', None)
        if kwargs:
            raise TypeError("Got unexpected keyword arguments.")

        # Load module
        mod = import_module("tnco_core.optimize.infinite_memory")

        # Load object
        obj = getattr(mod, 'Optimizer_' + cmodel.cost_type)

        # Save model
        self._cmodel = cmodel

        # Save cache
        self._ctree_cache = (ctree.dims, ctree._n_tensors, ctree._tensors_pos,
                             ctree._inds_order)

        # Get C++ core
        self._optimizer = obj(ctree,
                              cmodel.__get_core__(ctree._inds_order),
                              seed=seed,
                              disable_shared_inds=disable_shared_inds,
                              atol=atol,
                              min_ctree=_min_ctree)

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
    def disable_shared_inds(self):
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

    def is_valid(self, *, atol: float = 1e-5, return_message: str = False):
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

    def update(self, prob: BaseProbability) -> NoReturn:
        """Update 'Optimizer' using 'prob'.

        When 'Optimizer.update' is called, a potential move in the contraction
        tree is proposed. 'prob' must take the difference in cost and the
        original cost, and decide to either accept or reject the move.

        Args:
            prob: The probability to use to either accept or reject the
                proposed move.
        """
        self._optimizer.update(prob)

    @staticmethod
    def __build__(*args):
        (ctree, cmodel, prng_state, disable_shared_inds, min_ctree) = args
        return Optimizer(ctree,
                         cmodel,
                         seed=prng_state,
                         disable_shared_inds=disable_shared_inds,
                         _min_ctree=min_ctree)

    def __reduce__(self):
        return self.__build__, (self.ctree, self.cmodel, self.prng_state,
                                self.disable_shared_inds, self.min_ctree)

    def __repr__(self):
        return "Optimizer(ctree={}, cmodel={})".format(self.ctree, self.cmodel)

    def __eq__(self, other):
        return self.__reduce__()[1] == other.__reduce__()[1]
