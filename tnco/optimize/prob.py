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
from typing import Literal, Optional

__all__ = ['BaseProbability', 'Greedy', 'SimulatedAnnealing']


def BaseProbability(*,
                    cost_type: Optional[Literal['float32', 'float64',
                                                'float128',
                                                'float1024']] = 'float64'):
    """Acceptance probability.

    Always accept any proposed move.

    Args:
        cost_type: Type to use for the cost.
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
           cost_type: Optional[Literal['float32', 'float64', 'float128',
                                       'float1024']] = 'float64'):
    """Greedy probability.

    Accept a move only if the cost is not increasing.

    Args:
        cost_type: Type to use for the cost.
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
                       cost_type: Optional[Literal['float32', 'float64',
                                                   'float128',
                                                   'float1024']] = 'float64'):
    """Simulated Annealing.

    Accept a move using the Metropolis-Hastings probability.

    Args:
        cost_type: Type to use for the cost.
    """

    # Check cost_type
    cost_type = str(cost_type).lower()
    if cost_type not in ['float32', 'float64', 'float128', 'float1024']:
        raise NotImplementedError(f"'{cost_type=}' not available.")

    # Load module
    mod = import_module("tnco_core.optimize.prob")

    # Return object
    return getattr(mod, 'SimulatedAnnealing_' + cost_type)(beta)
