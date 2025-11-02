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
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple, Union

import autoray as ar
import more_itertools as mit
from rich.console import Console
from rich.progress import track

import tnco.utils.tensor as tensor_utils
import tnco.utils.tn as tn_utils
from tnco.ordered_frozenset import OrderedFrozenSet
from tnco.typing import Array, Index, Matrix, Qubit

__all__ = ['commute']


def commute(gate_A: Tuple[Matrix, Iterable[Qubit]],
            gate_B: Tuple[Matrix, Iterable[Qubit]],
            *,
            use_matrix_commutation: Optional[bool] = True,
            atol: Optional[float] = 1e-8) -> bool:
    """Check if the two gates commute.

    Check if the two gates commute.

    Args:
        gate_A: Gate to check commutation.
        gate_B: Gate to check commutation.
        use_matrix_commutation: If 'True', the commutation is checked by
            actually performing the commutation. Otherwise, only qubits are
            checked.
        atol: Absolute tollerance to check commutation.

    Returns:
        'True' is 'gate_A' and 'gate_B' commute. Othewise, return 'False'.
    """

    # Check if gate is valid
    def is_valid_gate(array, qubits):
        return len(qubits) > 0 and array.ndim == 2 and array.shape[
            0] == array.shape[1] and array.shape[0] == 2**len(
                qubits) and mit.all_unique(qubits)

    # Split
    array_A, qubits_A = gate_A
    array_B, qubits_B = gate_B
    array_A, array_B = map(lambda x: ar.do('asarray', x), (array_A, array_B))
    qubits_A, qubits_B = map(tuple, (qubits_A, qubits_B))

    # Check dimensions
    if not is_valid_gate(array_A, qubits_A):
        raise ValueError("'gate_A' is not valid.")
    if not is_valid_gate(array_B, qubits_B):
        raise ValueError("'gate_B' is not valid.")

    # Get all qubits
    all_qubits = tuple(mit.unique_everseen(qubits_A + qubits_B))

    # Get shared inds
    shared_qubits = frozenset(qubits_A) & frozenset(qubits_B)

    # If no qubit is shared, return true
    if not shared_qubits:
        return True

    # Exit if no matrix commutation is needed
    if not use_matrix_commutation:
        return False

    # Create contraction path
    def contraction_path_(qs_A, qs_B):
        xs_A = list(
            its.chain(
                map(lambda q: (q, 'i'), qs_A),
                map(lambda q: (q, 'shared'
                               if q in shared_qubits else 'f'), qs_A)))
        xs_B = list(
            its.chain(
                map(lambda q: (q, 'shared'
                               if q in shared_qubits else 'i'), qs_B),
                map(lambda q: (q, 'f'), qs_B)))
        xs_C = list(
            its.chain(map(lambda q: (q, 'i'), all_qubits),
                      map(lambda q: (q, 'f'), all_qubits)))
        return tensor_utils.get_einsum_path(xs_A, xs_B, xs_C)

    # Reshape unitaries
    array_A = array_A.reshape((2,) * 2 * len(qubits_A))
    array_B = array_B.reshape((2,) * 2 * len(qubits_B))

    # Compute AB and BA
    array_AB = ar.do('einsum', contraction_path_(qubits_A, qubits_B), array_A,
                     array_B)
    array_BA = ar.do('einsum', contraction_path_(qubits_B, qubits_A), array_B,
                     array_A)

    # Check commutation
    return ar.do('allclose', array_AB, array_BA, atol=atol)


def same(gate_A: Tuple[Matrix, Iterable[Qubit]],
         gate_B: Tuple[Matrix, Iterable[Qubit]],
         *,
         atol: Optional[float] = 1e-8) -> bool:
    """Check if two gates are the same.

    Check if two gates are the same.

    Args:
        gate_A: Gate to check.
        gate_B: Gate to check.
        atol: Absolute tollerance to check equality.

    Returns:
        'True' if 'gate_A' and 'gate_B' are the same. Otherwise, return
        'False'.
    """

    # Check if gate is valid
    def is_valid_gate(array, qubits):
        return len(qubits) > 0 and array.ndim == 2 and array.shape[
            0] == array.shape[1] and array.shape[0] == 2**len(
                qubits) and mit.all_unique(qubits)

    # Split
    array_A, qubits_A = gate_A
    array_B, qubits_B = gate_B
    array_A, array_B = map(lambda x: ar.do('asarray', x), (array_A, array_B))
    qubits_A, qubits_B = map(tuple, (qubits_A, qubits_B))

    # Check dimensions
    if not is_valid_gate(array_A, qubits_A):
        raise ValueError("'gate_A' is not valid.")
    if not is_valid_gate(array_B, qubits_B):
        raise ValueError("'gate_B' is not valid.")

    # If qubits differ, return false
    if len(qubits_A) != len(qubits_B) or any(
            map(lambda q: q not in qubits_A, qubits_B)):
        return False

    # Get new order for B and transpose it
    order = tuple(map(qubits_B.index, qubits_A))
    order += tuple(map(lambda x: x + len(qubits_A), order))
    array_B = array_B.reshape(
        (2,) * 2 * len(qubits_B)).transpose(order).reshape(
            (2**len(qubits_B), -1))

    # Get the all elements different from zero
    pos_A = ar.do('abs', array_A) > atol
    pos_B = ar.do('abs', array_B) > atol

    # If the location of elements different from zero is different, return False
    if not ar.do('allclose', pos_A, pos_B):
        return False

    # Check if all elements are the same, up to a constant
    W = array_A[pos_A].ravel() / array_B[pos_B].ravel()
    return ar.do('allclose', W, W[0], atol=atol)


@fts.singledispatch
def load(circuit: Iterable[Tuple[Matrix, Tuple[Qubit]]],
         *,
         initial_state: Optional[Union[str, Dict[Qubit, Matrix], None]] = '0',
         final_state: Optional[Union[str, Dict[Qubit, Matrix], None]] = '0',
         simplify: Optional[bool] = True,
         use_matrix_commutation: Optional[bool] = True,
         decompose_hyper_inds: Optional[bool] = True,
         fuse: Optional[float] = 4,
         atol: Optional[float] = 1e-8,
         backend: Optional[str] = None,
         seed: Optional[int] = None,
         verbose: Optional[int] = False,
         **kwargs) -> Tuple[List[Array], List[Tuple[Index]], FrozenSet[Index]]:
    """Load circuit and convert it to a list of tensors.

    Load circuit and convert it to a list of tensors.

    Args:
        circuit: List of gates.
        initial_state: Initial state state to apply to the circuit. If a 'dict'
            is used, qubits are the keys and the corresponding values can be
            either a single char token between '01+-', or a 1x2 matrix. If a
            qubit is missing, it is considered open. If a single token / matrix
            is used, the same is applied to all qubits. If 'None', all qubits
            are open.
        final_state: Final state state to apply to the circuit. If a 'dict' is
            used, qubits are the keys and the corresponding values can be
            either a single char token between '01+-', or a 1x2 matrix. If a
            qubit is missing, it is considered open. If a single token / matrix
            is used, the same is applied to all qubits. If 'None', all qubits
            are open.
        simplify: If 'True', gates that cancel each other will be simplified.
        use_matrix_commutation: If 'True', gates will be commuted by performing
            the actual matrix commutation while simplifying the circuit.
        decompose_hyper_inds: If 'True', decompose gates to get hyper-indices.
        fuse: Fuse tensors together up a width smaller than 'fuse'.  The width
            is defined as sum of the logarithms of all the dimensions of a
            given tensor.  Tensors are contracted so that
        atol: Absolute tollerance for array comparison.
        backend: Backend to use to fuse gates. See: `autoray.do`.
        seed: Seed for the random number generator.
        verbose: Verbose output.

    Returns:
        It returns a tuple of three elements:
            arrays: Arrays associated to each tensor.
            ts_inds: Indices associated to each tensor.
            output_inds: Output indices.
    """
    # Short names
    same_ = fts.partial(same, atol=atol)
    commute_ = fts.partial(commute,
                           use_matrix_commutation=use_matrix_commutation,
                           atol=atol)

    # Get all qubits
    circuit = tuple(circuit)
    qubits = kwargs.pop(
        '_qubits', OrderedFrozenSet(mit.flatten(map(op.itemgetter(1),
                                                    circuit))))

    if kwargs:
        raise TypeError('Got unexpected keyword argument(s).')

    # Collect gates
    all_gates = []

    # Any change?
    changes = False

    # For each gate ...
    for gate_ in track(circuit,
                       console=Console(stderr=True),
                       disable=(verbose <= 0),
                       transient=True,
                       description="Building TN..."):

        # Get adjoint
        gate_adj_ = (ar.do('asarray', gate_[0]).conj().T, gate_[1])

        # Check if there is at least one gate that can be simplifies
        pos_, status_ = next(
            filter(
                lambda x: x[1] is not None,
                its.starmap(
                    lambda i, gate:
                    (i, True if same_(gate, gate_adj_) else False
                     if not commute_(gate, gate_) else None),
                    enumerate(all_gates))),
            (None, False)) if simplify else (None, False)

        # If there is one, remove it
        if status_:
            del all_gates[pos_]
            changes = True

        # Otherwise, append it
        else:
            all_gates.insert(0, gate_)

    # If there are changes, try again
    if changes:
        return load(all_gates[::-1],
                    initial_state=initial_state,
                    final_state=final_state,
                    simplify=simplify,
                    use_matrix_commutation=use_matrix_commutation,
                    decompose_hyper_inds=decompose_hyper_inds,
                    fuse=fuse,
                    atol=atol,
                    seed=seed,
                    verbose=verbose,
                    _qubits=qubits)

    # Build TN
    qubit_map = defaultdict(int)
    arrays = []
    ts_inds = []
    for array_, qs_ in all_gates[::-1]:
        # Get qubits
        qs_ = tuple(map(lambda q: (q, qubit_map[q]), qs_))

        # Get tensor
        arrays.append(array_.reshape((2,) * 2 * len(qs_)))
        ts_inds.append(tuple(its.starmap(lambda q, x: (q, x + 1), qs_)) + qs_)

        # Update map
        for q_, _ in qs_:
            qubit_map[q_] += 1

    # Get open inds
    output_inds = OrderedFrozenSet(map(lambda x: (*x,),
                                       qubit_map.items())).union(
                                           map(lambda q: (q, 0), qubits))

    def get_state(state):
        valid_token = {
            '0': [1, 0],
            '1': [0, 1],
            '+': [1 / math.sqrt(2), 1 / math.sqrt(2)],
            '-': [1 / math.sqrt(2), -1 / math.sqrt(2)]
        }
        if state is None:
            return {}

        if isinstance(state, str):
            if state not in valid_token:
                raise ValueError(f"'{state}' is not valid.")
            return dict(zip(qubits, its.repeat(valid_token[state])))

        if not all(
                map(
                    lambda x: (isinstance(x, str) and len(x) == 1) or
                    (len(x) == 2 and abs(abs(x[0])**2 + abs(x[1])**2 - 1) < 1e-8
                    ), state.values())):
            raise ValueError("State is not valid.")

        return dict(
            filter(
                lambda x: x[0] in qubits,
                its.starmap(
                    lambda q, x:
                    (q, valid_token[x] if isinstance(x, str) else x),
                    state.items())))

    # Convert initial / final state
    initial_state = get_state(initial_state)
    final_state = get_state(final_state)
    #
    initial_state = list(
        filter(
            None,
            map(
                lambda q: (initial_state[q], ((q, 0),))
                if q in initial_state else None, qubits)))
    final_state = list(
        filter(
            None,
            its.starmap(
                lambda q, x: (ar.do('asarray', final_state[q]).conj(),
                              ((q, x),)) if q in final_state else None,
                mit.map_reduce(output_inds, lambda x: x[0], lambda x: x[1],
                               max).items())))

    # Attach initial / final state to tn
    if initial_state or final_state:
        arrays_, ts_inds_ = mit.transpose(initial_state + final_state)
        arrays.extend(arrays_)
        ts_inds.extend(ts_inds_)
        output_inds = output_inds.difference(mit.flatten(ts_inds_))

    # Decompose hyper inds if needed
    if decompose_hyper_inds:
        arrays, ts_inds, hyper_inds_map = tn_utils.decompose_hyper_inds(
            arrays, ts_inds, atol=atol)
        output_inds = frozenset(map(hyper_inds_map.get, output_inds))

    # Keep only the output inds that are still present in the tn
    output_inds = output_inds.intersection(mit.flatten(ts_inds))

    # Fuse if needed
    if fuse is not None and fuse > 0:
        # Find contraction path to fuse tensors
        path = tn_utils.fuse(ts_inds,
                             2,
                             max_width=fuse,
                             output_inds=output_inds,
                             seed=seed,
                             verbose=verbose)

        # Fuse tensors
        ts_inds, output_inds, arrays = tn_utils.contract(path,
                                                         ts_inds,
                                                         output_inds,
                                                         arrays,
                                                         backend=backend,
                                                         verbose=(verbose - 1))

    # Create tensors
    return arrays, ts_inds, frozenset(output_inds)


# Try to load cirq
try:
    import cirq

    @load.register
    def _(circuit: cirq.Circuit, *args, **kwargs):
        # Define gates to ignore
        def ignore_gate(gate):
            # Ignore measurement gates
            if cirq.is_measurement(gate):
                return True

            return False

        # Create circuit
        circuit = map(
            lambda g: (cirq.unitary(g), g.qubits),
            filter(lambda gate: not ignore_gate(gate),
                   circuit.all_operations()))

        # Load TN
        return load(circuit, *args, **kwargs)

    @load.register
    def _(circuit: cirq.Moment, *args, **kwargs):
        # Create circuit
        circuit = map(lambda g: (cirq.unitary(g), g.qubits), circuit)

        # Load TN
        return load(circuit, *args, **kwargs)

except ModuleNotFoundError:
    pass

# Try to load qiskit
try:
    import qiskit

    @load.register
    def _(circuit: qiskit.QuantumCircuit, *args, **kwargs):
        # Create circuit
        circuit = map(lambda g: (g.matrix, g.qubits), circuit)

        # Load TN
        return load(circuit, *args, **kwargs)

except ModuleNotFoundError:
    pass
