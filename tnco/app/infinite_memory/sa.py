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

import json
from dataclasses import dataclass
from random import Random
from sys import stderr
from time import perf_counter
from typing import Any, Iterable, List, Optional, Tuple, Union

import more_itertools as mit

import tnco.utils.tn as tn_utils
from tnco.app.app import BaseContractionResults, BaseOptimizer
from tnco.app.app import JSONEncoder as BaseJSONEncoder
from tnco.ctree import ContractionTree
from tnco.optimize.infinite_memory import Optimizer as OptimizerImpl
from tnco.optimize.infinite_memory.cost_model import SimpleCostModel
from tnco.optimize.prob import SimulatedAnnealing
from tnco.parallel import Parallel


class JSONEncoder(BaseJSONEncoder):
    """Helper for JSONEncoder.

    Helper for JSONEncoder.
    """

    def default(self, obj) -> Any:
        """Encode objects to JSON.

        Extend functionality of JSON.

        Args:
            obj: The object to encode.

        Returns:
            Encoded object in the JSON format.
        """
        if isinstance(obj, ContractionResults):
            return dict(**BaseJSONEncoder().default(obj),
                        disconnected_paths=obj.disconnected_paths)
        if hasattr(obj, 'to_json'):
            return obj.to_json()

        return super().default(obj)


@dataclass(repr=False, frozen=True, eq=False)
class ContractionResults(BaseContractionResults):
    """Contraction results.

    Contraction results for simulated annealing with no memory constraints.

    Args:
        disconnected_costs: The number of operations to perform to contract the
            tensor network for each disconnected path.
        disconnected_paths: The number of total disconnected paths is equal to
            the number of connected components in the optimized tensor network.
            Each path is in the SSA format and assume that each path is run
            independently using all the tensors in the original tensor network.
    """
    disconnected_costs: List[float]
    disconnected_paths: List[List[Tuple[int, int]]]

    def to_json(self) -> Any:
        """Return JSON.

        Return the contraction results using the JSON encoding.

        Returns:
            The encoded results in the JSON format.
        """
        return json.dumps(self, cls=JSONEncoder)


class Optimizer(BaseOptimizer):
    """Optimize tensor network with simulated annealing (infinite memory)

    Optimize the tensor network using simulated annealing and assuming infinite
    memory.
    """

    def optimize(self,
                 tn: Any,
                 betas: Union[Tuple[float, float], Iterable[float]],
                 n_steps: Optional[int] = None,
                 n_runs: Optional[int] = 1,
                 n_projs: Optional[int] = None,
                 timeout: Optional[float] = None,
                 **load_tn_options) -> Any:
        """Optimize the tensor network 'tn'.

        Optimize the tensor network 'tn'.

        Args:
            tn: The tensor network to optimize. Multiple formats are valid.
                (See: 'tnco.app.load_tn' for all the valid options)
            betas: The inverse temperature to use in the optimization. If
                'betas' is a tuple of two elements (and the number of steps
                'n_steps' is provided), the inverve temperature beta will be
                linearly interpolated between 'betas[0]' and 'betas[1]'.
                Otherwise, 'betas' will be used as inverse temperatures.
            n_steps: Number of steps for the optimization. It must be provided
                if 'betas' is a tuple of float representing the initial and
                final inverse temperature.
            n_runs: Number of indipendent runs to perform.
            n_projs: If 'tn' has sparse indices, the total number of
                projections to consider among all the sparse indices.
            timeout: If provided, stop optimzation after 'timeout' seconds.

        Returns:
            See: 'tnco.app.dump_results'

        Raises:
            ValueError: If parameters are not valid.
        """
        # Load the tensor network
        tn = self._load_tn(tn, **load_tn_options)

        # Initialize random generator
        rng = Random(self.seed)

        # Check 'n_steps'
        if n_steps is not None:
            if int(n_steps) != n_steps or n_steps <= 0:
                raise ValueError("'n_steps' must be a positive number.")
            n_steps = int(n_steps)

        # Expand betas
        if isinstance(betas, tuple) and len(betas) == 2:
            if n_steps is None:
                raise ValueError("'n_steps' must be provided if 'betas' "
                                 "has the format '(beta_min, beta_max)'.")
            if betas[0] == betas[1]:
                raise ValueError(
                    "'betas' must use the format '(beta_ini, beta_end)', "
                    "with 'beta_ini != beta_end'.")
            betas = mit.numeric_range(betas[0], betas[1],
                                      (betas[1] - betas[0]) / n_steps)

        # Get contraction cost
        cmodel = SimpleCostModel(cost_type=self.cost_type,
                                 sparse_inds=tn.sparse_inds,
                                 n_projs=n_projs)

        # Get probability
        prob = SimulatedAnnealing(cost_type=self.cost_type)

        def core_(seed, *, idx, status, stop, log2_total_cost):
            # Initialize results
            results = dict(disconnected_costs=[],
                           disconnected_paths=[],
                           runtime_s=[])

            # Get random contraction path
            for path in tn_utils.get_random_contraction_path(tn.ts_inds,
                                                             merge_paths=False,
                                                             seed=seed):

                # Get contraction tree
                ctree = ContractionTree(path,
                                        tn.ts_inds,
                                        tn.dims,
                                        output_inds=tn.output_inds,
                                        check_shared_inds=True)

                # Get optimizer
                opt = OptimizerImpl(ctree, cmodel, seed=seed)

                # Start clock
                runtime = perf_counter()

                # For each temperature ...
                for n, beta in enumerate(betas):
                    # Break if it has reached the number of steps
                    if n == n_steps or stop[idx]:
                        break

                    prob.beta = beta
                    opt.update(prob)

                    # Update status
                    status[idx] = n / len(betas)
                    log2_total_cost[idx] = opt.log2_min_total_cost

                # Stop clock
                runtime = perf_counter() - runtime

                # Append to results
                results['disconnected_costs'].append(opt.min_total_cost)
                results['disconnected_paths'].append(opt.min_ctree.path())
                results['runtime_s'].append(runtime)

            # Get total cost
            results['cost'] = sum(results['disconnected_costs'])

            # Get total runtime
            results['runtime_s'] = sum(results['runtime_s'])

            # Get disconnected paths
            if not results['disconnected_paths']:
                results['disconnected_paths'] = [()] * len(tn)

            # Get full path
            results['path'] = tn_utils.merge_contraction_paths(
                len(tn), results['disconnected_paths'])

            # Return results
            return ContractionResults(**results)

        # Get seeds
        seeds = rng.choices(range(2**32), k=n_runs)

        if self.verbose == 1:
            print("# Optimizing ...", file=stderr, flush=True, end='')

        # Run simulations
        results = Parallel(
            core_,
            seed=seeds,
            n_jobs=self.n_jobs,
            timeout=timeout,
            description="Optimizing...",
            text="[red]LOG2(COST)={task.fields[log2_total_cost]:1.2f}",
            buffers=[('log2_total_cost', 'f')],
            verbose=self.verbose - 1)

        if self.verbose == 1:
            print(" Done!", file=stderr, flush=True)

        # Get contraction results
        return self._dump_results(tn, sorted(results))
