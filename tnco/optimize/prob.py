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
"""Probability functions for optimization."""

from importlib import import_module
from typing import Literal
from warnings import warn

__all__ = [
    'BaseProbability', 'Greedy', 'SimulatedAnnealing', 'MetropolisHastings'
]


def BaseProbability(*,
                    cost_type: Literal['float32', 'float64', 'float128',
                                       'float1024'] = 'float64'):
    """Factory for acceptance probability (always accept).

    Always accept any proposed move.

    Args:
        cost_type: Type to use for the cost.

    Returns:
        The probability object.

    Examples:
        >>> from tnco.optimize.prob import BaseProbability
        >>> prob = BaseProbability()
        >>> prob(10, 100)
        1.0
    """

    # Check cost_type
    cost_type = str(cost_type).lower()
    if cost_type not in ['float32', 'float64', 'float128', 'float1024']:
        raise NotImplementedError(f"'{cost_type=}' not available.")

    # Load module
    mod = import_module("tnco_core.optimize.prob")

    # Return object
    return getattr(mod, 'BaseProbability_' + cost_type)()


def Greedy(*,
           cost_type: Literal['float32', 'float64', 'float128',
                              'float1024'] = 'float64'):
    """Factory for greedy probability.

    Accept a move only if the cost is not increasing.

    Args:
        cost_type: Type to use for the cost.

    Returns:
        The probability object.

    Examples:
        >>> from tnco.optimize.prob import Greedy
        >>> prob = Greedy()
        >>> prob(-10, 100)
        1.0
        >>> prob(10, 100)
        0.0
    """

    # Check cost_type
    cost_type = str(cost_type).lower()
    if cost_type not in ['float32', 'float64', 'float128', 'float1024']:
        raise NotImplementedError(f"'{cost_type=}' not available.")

    # Load module
    mod = import_module("tnco_core.optimize.prob")

    # Return object
    return getattr(mod, 'Greedy_' + cost_type)()


def SimulatedAnnealing(beta: float = 0,
                       *,
                       cost_type: Literal['float32', 'float64', 'float128',
                                          'float1024'] = 'float64'):
    """Factory for Simulated Annealing (Deprecated).

    Accept a move using the Metropolis-Hastings probability. Deprecated in
    ``v0.2``, see ``MetropolisHastings``.


    Args:
        beta: Inverse temperature.
        cost_type: Type to use for the cost.

    Returns:
        The probability object.
    """
    warn(
        "'SimulatedAnnealing' is deprecated, and it will be "
        "removed in 'v0.2'. Please use 'MetropolisHastings'.",
        DeprecationWarning,
        stacklevel=2)

    # Return object
    return MetropolisHastings(beta, cost_type=cost_type)


def MetropolisHastings(beta: float = 0,
                       *,
                       cost_type: Literal['float32', 'float64', 'float128',
                                          'float1024'] = 'float64'):
    """Factory for Metropolis-Hastings probability.

    Accept a move using the Metropolis-Hastings probability.

    Args:
        beta: Inverse temperature.
        cost_type: Type to use for the cost.

    Returns:
        The probability object.

    Examples:
        >>> from tnco.optimize.prob import MetropolisHastings
        >>> prob = MetropolisHastings(beta=1)
        >>> prob(-10, 100)
        1.0
        >>> prob(10, 100) < 1
        True
    """

    # Check cost_type
    cost_type = str(cost_type).lower()
    if cost_type not in ['float32', 'float64', 'float128', 'float1024']:
        raise NotImplementedError(f"'{cost_type=}' not available.")

    # Load module
    mod = import_module("tnco_core.optimize.prob")

    # Return object
    return getattr(mod, 'MetropolisHastings_' + cost_type)(beta)
