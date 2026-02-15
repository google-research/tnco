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
from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple, Union

import autoray as ar
import more_itertools as mit
from rich.progress import Console, Progress, track

from tnco.app import Optimizer, Tensor, TensorNetwork
from tnco.app.app import BaseContractionResults
from tnco.typing import Array, Circuit, Matrix, Qubit
from tnco.utils.circuit import load
from tnco.utils.tn import contract, get_einsum_subscripts

__all__ = ['Sampler']


def is_classical_operation(m: Matrix) -> bool:
    """Check if given matrix is a classical operation.

    An operation is classical if ``m @ v`` only permutes the elements of ``v``
    (excluding a change of phase).

    Args:
        m: Matrix to check.

    Returns:
        ``True`` if ``m`` is a classical operation.
    """
    m = ar.do('asarray', m)
    if m.ndim != 2 or m.shape[0] != m.shape[1] and int(math.log2(
            m.shape[0])) != math.log2(m.shape[0]):
        return False

    # Get elements different from zero
    row_pos, col_pos = ar.do('where', m)

    # It should be a permutation of indices
    if not (sorted(row_pos) == sorted(col_pos) == list(range(m.shape[0]))):
        return False

    # All elements should be one, excluding a phase
    if not all(abs(m)[m != 0] == 1):
        return False

    # Excluding a phase change, it's a permutation
    return True


@dataclass(init=False, eq=False, repr=False, frozen=True)
class SamplingIntermediateState:
    """
    Store the intermediate state of the sampling routine.
    """
    data = Tuple[Union[Tuple[TensorNetwork, BaseContractionResults, Array,
                             Tuple[Qubit], Tuple[Qubit]],
                       Tuple[None, None, Array, None, Tuple[Qubit]]], ...]
    qubits = FrozenSet[Qubit]

    def __init__(self, data, qubits):
        object.__setattr__(self, 'data', tuple(data))
        object.__setattr__(self, 'qubits', frozenset(qubits))

    def __getitem__(self, k):
        return self.data[k]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


@fts.singledispatch
def sample(
    circuit: Union[Iterable[Tuple[Matrix, Tuple[Qubit]]],
                   SamplingIntermediateState],
    optimizer: Optimizer,
    n_samples: Optional[int] = 1,
    *,
    simplify: Optional[bool] = True,
    use_matrix_commutation: Optional[bool] = True,
    decompose_hyper_inds: Optional[bool] = True,
    fuse: Optional[float] = 4,
    qubit_order: Optional[Iterable[Qubit]] = None,
    normalize: Optional[bool] = True,
    return_intermediate_state_only: Optional[bool] = False,
    dtype: Optional[Any] = None,
    optimization_backend: Optional[str] = None,
    contraction_backend: Optional[str] = None,
    seed: Optional[int] = None,
    verbose: Optional[int] = False,
    **optimize_params
) -> Union[Tuple[Dict[str, int], Tuple[Qubit]], SamplingIntermediateState]:
    """Sample bitstrings from a circuit.

    Sample bitstrings from a circuit using the Bravyi-Gosset-Liu algorithm
    presented in "How to Simulate Quantum Measurement without Computing
    Marginals", Phys. Rev. Lett. 128, 220503 (2022).

    Args:
        circuit: The circuit to sample from. If a ``SamplingIntermediateState``
            is passed instead, the optimization phase of the circuit is skipped
            and the intermediate state is used to sample.
        optimizer: The ``Optimizer`` to use for the tensor network optimization.
        n_samples: The number of total samples to collect.
        simplify: If ``True``, gates that cancel each other will be simplified.
        use_matrix_commutation: If ``True``, gates will be commuted by
            performing the actual matrix commutation while simplifying the
            circuit.
        decompose_hyper_inds: If ``True``, decompose gates to get hyper-indices.
        fuse: Fuse tensors together up a width smaller than ``fuse``.  The width
            is defined as sum of the logarithms of all the dimensions of a
            given tensor.
        qubit_order: If provided, the order of qubits to use.
        normalize: If ``True``, the total number of hits are divided by the
            number of samples.
        optimize_params: The parameters to pass to
            ``tnco.app.Optimizer.optimize`` to optimize each partial
            contraction.
        return_intermediate_state_only: If ``True``, skip the sampling and
            return the ``SamplingIntermediateState``.
        dtype: The type to use for the contraction.
        optimization_backend: The backend to use to manipulate arrays while
            loading and optimizing the tensor network.
        contraction_backend: The backend to use for the contraction.
        seed: The seed to use for the sampling.
        verbose: Verbose output.

    Returns:
        If ``return_intermediate_state_only``, a ``SamplingIntermediateState``
        is returned to be reused multiple times. Otherwise, it returns a
        dictionary with the fraction of hits for each bitstring sampled, and
        the order of qubits. If ``normalize=False``, the number of hits is
        returned instead of its fraction.
    """

    # Short for asarray
    def asarray(*args, like=optimization_backend, **kwargs):
        return ar.do('asarray', *args, dtype=dtype, like=like, **kwargs)

    # Convert qubit_order to tuple
    qubit_order = None if qubit_order is None else tuple(qubit_order)

    # Skip if circuit is a 'SamplingIntermediateState'
    if not isinstance(circuit, SamplingIntermediateState):

        # Convert arrays
        circuit = list((asarray(m), qs) for m, qs in circuit)

        # Check if all operations are valid
        if not all(
                len(qs) == 1 or is_classical_operation(m) for m, qs in circuit):
            raise ValueError(
                "Only 1-qubit operations and linear transformations "
                "(with or without phase change) are allowed.")

        # Get all qubits
        qubits = frozenset(mit.flatten(map(op.itemgetter(1), circuit)))

        # Check consistency with qubit_order
        if qubit_order is not None and frozenset(qubit_order) != qubits:
            raise ValueError(
                "'qubit_order' is not consistent with qubits in 'circuit'.")

        # Load partial tensors
        partial_tn = list(
            ((m != 0).astype(int), None, None, q) if is_classical_operation(
                m) else load(circuit[:i + 1],
                             initial_state='0',
                             final_state=None,
                             simplify=simplify,
                             use_matrix_commutation=use_matrix_commutation,
                             decompose_hyper_inds=decompose_hyper_inds,
                             fuse=fuse) + (q,)
            for i, (m, q) in track(enumerate(circuit),
                                   total=len(circuit),
                                   description="Loading TNs...",
                                   disable=verbose <= 0,
                                   console=Console(stderr=True)))

        # Convert to tensors
        partial_tn = list(
            (None, arrays, None, op_qubits) if ts_inds is None else (
                TensorNetwork(map(fts.partial(Tensor, dims=2), ts_inds + list(
                    (x,)
                    for x in output_inds)),
                              output_inds=()), arrays,
                tuple(map(op.itemgetter(0), output_inds)), op_qubits)
            for arrays, ts_inds, output_inds, op_qubits in partial_tn)

        def optimize(tn):
            tn_, res = optimizer.optimize(tn,
                                          fuse=False,
                                          decompose_hyper_inds=False,
                                          **optimize_params)
            assert tn_ == tn

            # Return the optimization with the smallest cost
            return sorted(res, key=lambda x: x.cost)[0]

        partial_tn = SamplingIntermediateState(
            ((None, None, arrays, output_inds, op_qubits) if tn is None else
             (tn, optimize(tn),
              list(map(fts.partial(asarray, like=contraction_backend), arrays)),
              output_inds, op_qubits)
             for tn, arrays, output_inds, op_qubits in track(
                 partial_tn,
                 description="Optimizing TNs...",
                 disable=verbose <= 0,
                 console=Console(stderr=True))),
            qubits=qubits)
    else:
        partial_tn = circuit

    # Just return the partial state if needed
    if return_intermediate_state_only:
        return partial_tn

    # Initialize rng
    rng = Random(seed)

    # Get all qubits
    if qubit_order is not None:
        if frozenset(qubit_order) != partial_tn.qubits:
            raise ValueError(
                "'qubit_order' is not consistent with qubits in 'circuit'.")
        qubits = qubit_order
    else:
        qubits = tuple(partial_tn.qubits)
    n_qubits = len(qubits)

    # Initialize sampled bitstrings
    sampled_bitstrings = defaultdict(int)

    with Progress(disable=verbose <= 0, console=Console(stderr=True)) as pbar:

        task_1 = pbar.add_task("Sampling...", total=n_samples)

        for _ in range(n_samples):

            if verbose > 1:
                task_2 = pbar.add_task("Contracting...", total=len(partial_tn))

            # Initialize state
            bitstring = ar.do('zeros', n_qubits, dtype=int)

            # For each operation ...
            for n, (tn, contraction, arrays, output_qubits,
                    op_qubits) in enumerate(partial_tn):

                if tn is None:
                    # Get location of the qubits
                    qubits_loc = list(map(qubits.index, op_qubits))

                    # Initialize partial state
                    partial_bitstring = ar.do('zeros',
                                              2**len(op_qubits),
                                              dtype=int)
                    partial_bitstring[int(
                        ''.join(map(str, bitstring[qubits_loc])), 2)] = 1

                    # Get new state after apply the linera transformation
                    [partial_bitstring
                    ] = ar.do('where', (arrays @ partial_bitstring) % 2)[0]
                    partial_bitstring = list(
                        map(int,
                            bin(partial_bitstring)[2:].zfill(len(op_qubits))))

                    # Apply the new state
                    bitstring[qubits_loc] = partial_bitstring

                    # Go to the next operation
                    continue

                assert isinstance(tn, TensorNetwork)

                # Convert arrays
                arrays = list(
                    map(fts.partial(asarray, like=contraction_backend), arrays))

                # Get subscripts
                get_einsum_subscripts(tn.ts_inds)

                # Get path

                # Get location of the qubit (there should be only one)
                [qubit_loc] = map(qubits.index, op_qubits)

                # Get arrays with original x[qubit_loc]
                sub_bitstring_arrays = list(
                    asarray([0, 1] if bitstring[x] else [1, 0])
                    for x in map(qubits.index, output_qubits))

                # Compute probability
                _, _, [prob_0] = contract(contraction.path,
                                          tn.ts_inds,
                                          output_inds=(),
                                          arrays=arrays + sub_bitstring_arrays,
                                          backend=contraction_backend,
                                          verbose=verbose - 4)
                prob_0 = abs(prob_0)**2

                # Update arrays with x[qubit_loc] flipped
                sub_bitstring_arrays[output_qubits.index(
                    op_qubits[0])] = asarray(
                        [1, 0] if bitstring[qubit_loc] else [0, 1])

                # Compute probability
                _, _, [prob_1] = contract(contraction.path,
                                          tn.ts_inds,
                                          output_inds=(),
                                          arrays=arrays + sub_bitstring_arrays,
                                          backend=contraction_backend,
                                          verbose=verbose - 4)
                prob_1 = abs(prob_1)**2

                # Flip bit
                if rng.random() < prob_1 / (prob_0 + prob_1):
                    bitstring[qubit_loc] ^= 1

                # Update pbar
                if verbose > 1:
                    pbar.update(task_2, advance=1)

                # Go to the next operation
                continue

            # Add to the sampled bitstrings
            sampled_bitstrings[''.join(map(str, bitstring))] += 1

            # Update pbar
            pbar.update(task_1, advance=1)
            if verbose > 1:
                pbar.update(task_2, refresh=True)
                pbar.remove_task(task_2)

        # Finalize pbar
        pbar.update(task_1, refresh=True)

    # Normalize if needed
    if normalize:
        sampled_bitstrings = dict(
            its.starmap(lambda b, n: (b, n / n_samples),
                        sampled_bitstrings.items()))

    # Return sampled bitstrings
    return dict(
        sorted(sampled_bitstrings.items(), key=op.itemgetter(1),
               reverse=True)), qubits


# Try to load cirq
try:
    import cirq

    from tnco.utils.circuit import cirq_to_arrays

    @sample.register
    def _(circuit: cirq.AbstractCircuit, *args, **kwargs):
        return sample(
            cirq_to_arrays(circuit.all_operations(),
                           dtype=kwargs.get('dtype', None),
                           backend=kwargs.get('backend', None)), *args,
            **kwargs)

    @sample.register
    def _(circuit: cirq.Moment, *args, **kwargs):
        return sample(
            cirq_to_arrays(circuit,
                           dtype=kwargs.get('dtype', None),
                           backend=kwargs.get('backend', None)), *args,
            **kwargs)

except ModuleNotFoundError:
    pass

# Try to load qiskit
try:
    import qiskit

    from tnco.utils.circuit import qiskit_to_arrays

    @sample.register
    def _(circuit: qiskit.QuantumCircuit, *args, **kwargs):
        return load(
            qiskit_to_arrays(circuit,
                             dtype=kwargs.get('dtype', None),
                             backend=kwargs.get('backend', None)), *args,
            **kwargs)

except ModuleNotFoundError:
    pass


@dataclass(frozen=True)
class Sampler:
    """Sample bitstrings from a circuit.

    Sample bitstrings from a circuit using the Bravyi-Gosset-Liu algorithm
    presented in "How to Simulate Quantum Measurement without Computing
    Marginals", Phys. Rev. Lett. 128, 220503 (2022).

    Args:
        max_width: Maximum width to use. The width is defined as sum of the
            logarithms of all the dimensions of a given tensor. Tensors are
            contracted so that the width of the contracted tensor is smaller
            than ``max_width``.
        n_jobs: Number of processes to use. By default, all available cores are
            used. If ``n_jobs`` is a positive number, ``n_jobs`` processes will
            be used. If ``n_jobs`` is negative, ``n_cpus + n_jobs + 1`` will be
            used. If ``n_jobs`` is zero, it will raise a ``ValueError``. (See:
            ``tnco.parallel.Parallel``)
        width_type: The type to use to represent the width. (See:
            ``tnco.optimize.finite_width.cost_model.SimpleCostModel``)
        cost_type: The type to use to represent the cost. (See:
            ``tnco.optimize.finite_width.cost_model.SimpleCostModel``)
        atol: Absolute tolerance when checking for hyper-indices.
        dtype: The type to use for the arrays.
        optimization_backend: The backend to use to manipulate arrays while
            loading and optimizing the tensor network.
        seed: Seed to use.
        verbose: Verbose output.
    """
    max_width: Optional[float] = None
    n_jobs: Optional[int] = -1
    width_type: Optional[str] = 'float32'
    cost_type: Optional[str] = 'float64'
    atol: Optional[float] = 1e-5
    dtype: Optional[Any] = None
    optimization_backend: Optional[str] = None
    seed: Optional[int] = None
    verbose: Optional[int] = False

    def __post_init__(self):
        # Get rng
        object.__setattr__(self, '_rng', Random(self.seed))

        # Get optimizer
        optimizer = Optimizer(max_width=self.max_width,
                              n_jobs=self.n_jobs,
                              width_type=self.width_type,
                              cost_type=self.cost_type,
                              atol=self.atol,
                              dtype=self.dtype,
                              backend=self.optimization_backend,
                              seed=self._rng.randrange(2**32),
                              verbose=self.verbose - 5)
        object.__setattr__(self, '_optimizer', optimizer)

        # Slices are not yet supported
        if self.max_width is not None and self.max_width < float('inf'):
            raise NotImplementedError(
                "Sampling with finite width is not yet implemented.")

    def sample(
        self,
        circuit: Union[Circuit, SamplingIntermediateState],
        n_samples: Optional[int] = 1,
        *,
        simplify: Optional[bool] = True,
        use_matrix_commutation: Optional[bool] = True,
        decompose_hyper_inds: Optional[bool] = True,
        fuse: Optional[float] = 4,
        qubit_order: Optional[Iterable[Qubit]] = None,
        normalize: Optional[bool] = True,
        return_intermediate_state_only: Optional[bool] = False,
        contraction_backend: Optional[str] = None,
        **optimize_params
    ) -> Union[Tuple[Dict[str, int], Tuple[Qubit]], SamplingIntermediateState]:
        """Sample bitstrings from a circuit.

        Sample bitstrings from a circuit using the Bravyi-Gosset-Liu algorithm
        presented in "How to Simulate Quantum Measurement without Computing
        Marginals", Phys. Rev. Lett. 128, 220503 (2022).

        Args:
            circuit: Circuit to sample from. If a ``SamplingIntermediateState``
                is passed instead, the optimization phase of the circuit is
                skipped and the intermediate state is used to sample.
            n_samples: The number of total samples to collect.
            simplify: If ``True``, gates that cancel each other will be
                simplified.
            use_matrix_commutation: If ``True``, gates will be commuted by
                performing the actual matrix commutation while simplifying the
                circuit.
            decompose_hyper_inds: If ``True``, decompose gates to get
                hyper-indices.
            fuse: Fuse tensors together up a width smaller than ``fuse``.  The
                width is defined as sum of the logarithms of all the dimensions
                of a given tensor.
            qubit_order: If provided, the order of qubits to use.
            normalize: If ``True``, the total number of hits are divided by the
                number of samples.
            return_intermediate_state_only: If ``True``, skip the sampling and
                return the ``SamplingIntermediateState``.
            contraction_backend: The backend to use for the contraction.
            **optimize_params: Parameters to use to optimize the tensor network
                contraction. (See ``tnco.app.Optimizer.optimize``)

        Returns:
            If ``return_intermediate_state_only``, a
            ``SamplingIntermediateState`` is returned to be reused multiple
            times. Otherwise, it returns a dictionary with the fraction of hits
            for each bitstring sampled, and the order of qubits. If
            ``normalize=False``, the number of hits is returned instead of its
            fraction.
        """

        return sample(
            circuit,
            optimizer=self._optimizer,
            n_samples=n_samples,
            simplify=simplify,
            use_matrix_commutation=use_matrix_commutation,
            decompose_hyper_inds=decompose_hyper_inds,
            fuse=fuse,
            qubit_order=qubit_order,
            normalize=normalize,
            return_intermediate_state_only=return_intermediate_state_only,
            dtype=self.dtype,
            optimization_backend=self.optimization_backend,
            contraction_backend=contraction_backend,
            seed=self._rng.randrange(2**32),
            verbose=self.verbose,
            **optimize_params)
