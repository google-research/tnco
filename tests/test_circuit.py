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

import itertools as its
from math import sqrt
from os import environ
from random import Random

import cirq
import more_itertools as mit
import numpy as np
import pytest
import qiskit
from qiskit.circuit.random import random_circuit
from quimb.tensor import Tensor, TensorNetwork
from tnco.utils.circuit import commute, load, same

# Initialize RNG
rng = Random(
    environ.get('PYTEST_SEED') +
    environ.get('PYTEST_XDIST_WORKER') if 'PYTEST_SEED' in environ else None)

# Fix max number of repetitions
max_repeat = max(1, float(environ.get('PYTEST_MAX_REPEAT', 'inf')))

# Fix ratio of number of tests
fraction_n_tests = max(
    min(float(environ.get('PYTEST_FRACTION_N_TESTS', '1')), 1), 0)


def repeat(n: int):
    return pytest.mark.repeat(max(min(n * fraction_n_tests, max_repeat), 1))


@pytest.fixture
def random_seed():
    return rng.randrange(2**32)


@repeat(20)
def test_Commute(random_seed, **kwargs):
    # Commutation using cirq
    def commute_(gate_A, gate_B, *, use_matrix_commutation=True, atol=1e-5):
        if use_matrix_commutation:
            gate_A = cirq.MatrixGate(gate_A[0]).on(*gate_A[1])
            gate_B = cirq.MatrixGate(gate_B[0]).on(*gate_B[1])
            AB = cirq.unitary(cirq.Circuit([gate_A, gate_B]))
            BA = cirq.unitary(cirq.Circuit([gate_B, gate_A]))
            return np.allclose(AB, BA, atol=atol)
        return not frozenset(gate_A[1]) & frozenset(gate_B[1])

    # Initialize RNG
    rng = Random(random_seed)

    # Set parameters
    n_qubits = kwargs.get('n_qubits', 5)
    n_moments = kwargs.get('n_moments', 5)

    # Get a random circuit
    circuit = tuple(
        map(
            lambda gate: (cirq.unitary(gate), gate.qubits),
            cirq.testing.random_circuit(
                n_qubits, n_moments, op_density=1,
                random_state=random_seed).all_operations()))

    # Check commutation
    assert all(
        map(
            lambda gs: commute_(*gs, use_matrix_commutation=False) == commute(
                *gs, use_matrix_commutation=False),
            zip(rng.choices(circuit, k=200), rng.choices(circuit, k=200))))
    assert all(
        map(
            lambda gs: commute_(*gs, use_matrix_commutation=True) == commute(
                *gs, use_matrix_commutation=True),
            zip(rng.choices(circuit, k=200), rng.choices(circuit, k=200))))


@repeat(20)
def test_Same(random_seed, **kwargs):
    # Same using quimb
    def same_(gate_A, gate_B, atol=1e-5):
        gate_A = Tensor(
            gate_A[0].reshape((2,) * 2 * len(gate_A[1])),
            its.chain(zip(gate_A[1], its.repeat('f')),
                      zip(gate_A[1], its.repeat('i'))))
        gate_B = Tensor(
            gate_B[0].reshape((2,) * 2 * len(gate_B[1])),
            its.chain(zip(gate_B[1], its.repeat('f')),
                      zip(gate_B[1], its.repeat('i'))))
        return frozenset(gate_A.inds) == frozenset(gate_B.inds) and np.allclose(
            gate_A.data, gate_B.transpose_like(gate_A).data, atol=1e-5)

    # Initialize RNG
    rng = Random(random_seed)

    # Set parameters
    n_qubits = kwargs.get('n_qubits', 5)
    n_moments = kwargs.get('n_moments', 5)

    # Get a random circuit
    circuit = tuple(
        map(
            lambda gate: (cirq.unitary(gate), gate.qubits),
            cirq.testing.random_circuit(
                n_qubits, n_moments, op_density=1,
                random_state=random_seed).all_operations()))

    # Check same
    assert all(
        map(lambda gs: same_(*gs) == same(*gs),
            zip(rng.choices(circuit, k=200), rng.choices(circuit, k=200))))

    assert all(
        map(lambda g: same_(g, g) and same(g, g), rng.choices(circuit, k=200)))


@repeat(20)
def test_LoadArbitraryInitialFinalState(random_seed, **kwargs):
    # Get rng
    rng = Random(random_seed)
    np_rng = np.random.default_rng(random_seed)

    # Get parameters
    n_qubits = kwargs.get('n_qubits', rng.randint(5, 10))
    n_moments = kwargs.get('n_moments', rng.randint(10, 20))
    op_density = kwargs.get('op_density', 1)
    simplify = kwargs.get('simplify', rng.choice([False, True]))
    decompose_hyper_inds = kwargs.get('decompose_hyper_inds',
                                      rng.choice([False, True]))
    fuse = kwargs.get('fuse', 4 * rng.random())
    use_matrix_commutation = kwargs.get('use_matrix_commutation',
                                        rng.choice([False, True]))
    verbose = kwargs.get('verbose', False)

    # Create a random initial / final state
    def random_qubit():
        if rng.randrange(2):
            return None

        if rng.randrange(2):
            x = np_rng.normal(size=2) + 1j * np_rng.normal(size=2)
            x /= np.linalg.norm(x)
            return x

        return '01+-'[rng.randrange(4)]

    # Get random circuit
    if rng.randrange(2):
        circuit = cirq.testing.random_circuit(n_qubits,
                                              n_moments,
                                              op_density=op_density,
                                              random_state=random_seed)
    else:
        circuit = cirq.testing.random_circuit(n_qubits,
                                              n_moments,
                                              op_density=op_density,
                                              gate_domain={
                                                  cirq.ZPowGate(): 1,
                                                  cirq.ZZPowGate(): 2,
                                                  cirq.CZPowGate(): 2,
                                                  cirq.CCZPowGate(): 3,
                                                  cirq.ISWAP: 2
                                              },
                                              random_state=random_seed)
    circuit = map((cirq.H**0.5).on, circuit.all_qubits()) + circuit + map(
        (cirq.H**0.5).on, circuit.all_qubits())

    # Get qubits
    qubits = sorted(circuit.all_qubits())

    # Get initial / final state
    initial_state = '01+-'[rng.randrange(4)] if rng.randrange(2) else dict(
        filter(lambda x: x[1] is not None,
               zip(qubits, mit.repeatfunc(random_qubit, n_qubits))))
    final_state = '01+-'[rng.randrange(4)] if rng.randrange(2) else dict(
        filter(lambda x: x[1] is not None,
               zip(qubits, mit.repeatfunc(random_qubit, n_qubits))))

    # Load arrays
    arrays, ts_inds, output_inds = load(
        circuit,
        initial_state=initial_state,
        final_state=final_state,
        verbose=verbose,
        simplify=simplify,
        decompose_hyper_inds=decompose_hyper_inds,
        fuse=fuse,
        use_matrix_commutation=use_matrix_commutation,
        seed=random_seed)

    def get_state(x):
        valid_token = {
            '0': [1, 0],
            '1': [0, 1],
            '+': [1 / sqrt(2), 1 / sqrt(2)],
            '-': [1 / sqrt(2), -1 / sqrt(2)]
        }
        return valid_token[x] if isinstance(x, str) else x

    # Get exact unitary
    ex = Tensor(
        cirq.unitary(circuit).reshape((2,) * 2 * n_qubits),
        list(map(lambda q:
                 (q, 'f'), qubits)) + list(map(lambda q: (q, 'i'), qubits)))
    if isinstance(initial_state, str):
        ex &= list(
            map(Tensor, its.repeat(get_state(initial_state)),
                map(lambda q: ((q, 'i'),), qubits)))
    else:
        ex &= list(
            map(Tensor, map(get_state, initial_state.values()),
                map(lambda q: ((q, 'i'),), initial_state.keys())))
    if isinstance(final_state, str):
        ex &= list(
            map(Tensor, its.repeat(get_state(final_state)),
                map(lambda q: ((q, 'f'),), qubits)))
    else:
        ex &= list(
            map(
                lambda x: x.conj(),
                map(Tensor, map(get_state, final_state.values()),
                    map(lambda q: ((q, 'f'),), final_state.keys()))))

    # Check if there are open inds
    outer_inds = ex.outer_inds()

    # Contract
    ex = ex.contract()
    ts = TensorNetwork(map(Tensor, arrays,
                           ts_inds)).contract(output_inds=output_inds)

    # Transpose if needed
    if outer_inds:
        ts = ts.reindex(
            dict(map(lambda x: (x, (x[0], 'i' if x[1] == 0 else 'f')),
                     ts.inds))).transpose_like(ex)

    # Check
    ex = getattr(ex, 'data', ex)
    ts = getattr(ts, 'data', ts)
    np.testing.assert_allclose(ex, ts, atol=1e-5)


@repeat(20)
def test_LoadUnitary(random_seed, **kwargs):
    # Get rng
    rng = Random(random_seed)

    # Get parameters
    n_qubits = kwargs.get('n_qubits', rng.randint(5, 10))
    n_moments = kwargs.get('n_moments', rng.randint(10, 20))
    op_density = kwargs.get('op_density', 1)
    simplify = kwargs.get('simplify', rng.choice([False, True]))
    decompose_hyper_inds = kwargs.get('decompose_hyper_inds',
                                      rng.choice([False, True]))
    fuse = kwargs.get('fuse', 4 * rng.random())
    use_matrix_commutation = kwargs.get('use_matrix_commutation',
                                        rng.choice([False, True]))
    verbose = kwargs.get('verbose', False)

    # Get random circuit
    if rng.randrange(2):
        circuit = cirq.testing.random_circuit(n_qubits,
                                              n_moments,
                                              op_density=op_density,
                                              random_state=random_seed)
    else:
        circuit = cirq.testing.random_circuit(n_qubits,
                                              n_moments,
                                              op_density=op_density,
                                              gate_domain={
                                                  cirq.ZPowGate(): 1,
                                                  cirq.ZZPowGate(): 2,
                                                  cirq.CZPowGate(): 2,
                                                  cirq.CCZPowGate(): 3,
                                                  cirq.ISWAP: 2
                                              },
                                              random_state=random_seed)
    circuit = map((cirq.H**0.5).on, circuit.all_qubits()) + circuit + map(
        (cirq.H**0.5).on, circuit.all_qubits())

    # Load arrays
    arrays, ts_inds, output_inds = load(
        circuit,
        initial_state=None,
        final_state=None,
        verbose=verbose,
        simplify=simplify,
        decompose_hyper_inds=decompose_hyper_inds,
        fuse=fuse,
        use_matrix_commutation=use_matrix_commutation,
        seed=random_seed)

    # Get exact unitary
    U = cirq.unitary(circuit)

    # Contract arrays
    tn = TensorNetwork(its.starmap(Tensor, zip(
        arrays, ts_inds))).contract(output_inds=output_inds).reindex(
            dict(
                its.starmap(lambda x, p: ((x, p), (x, 'i' if p == 0 else 'f')),
                            output_inds)))
    tn = tn.fuse(
        dict(I=sorted(zip(circuit.all_qubits(), its.repeat('i'))),
             F=sorted(zip(circuit.all_qubits(),
                          its.repeat('f'))))).transpose(*'FI')

    # Check
    np.testing.assert_allclose(tn.data, U, atol=1e-5)

    # Get circuit with commutable gates only
    circuit = cirq.testing.random_circuit(10,
                                          20,
                                          op_density=1,
                                          gate_domain={
                                              cirq.ZPowGate(): 1,
                                              cirq.ZZPowGate(): 2,
                                              cirq.CZPowGate(): 2,
                                              cirq.CCZPowGate(): 3,
                                          },
                                          random_state=None)
    # Shuffle circuit
    circuit = list((circuit + cirq.inverse(circuit)).all_operations())
    circuit = cirq.Circuit(rng.sample(circuit, k=len(circuit)))

    # Load circuit
    arrays, ts_inds, output_inds = load(circuit,
                                        initial_state=None,
                                        final_state=None,
                                        simplify=True,
                                        fuse=0,
                                        decompose_hyper_inds=False)

    # Everything should be empty
    assert not arrays and not ts_inds and not output_inds


@repeat(20)
def test_LoadQisKit(random_seed):
    # Set number of qubits and depth of the circuit
    n_qubits = 6
    circuit_depth = 12

    # Implement sqrt of Hadamard
    sqrt_H = qiskit.circuit.library.UnitaryGate(cirq.unitary(cirq.H**0.5),
                                                label='√H')

    # Get a random circuit
    circuit = qiskit.QuantumCircuit(n_qubits)
    mit.consume(map(lambda i: circuit.append(sqrt_H, [i]), range(n_qubits)))
    circuit = circuit.compose(
        random_circuit(n_qubits, circuit_depth, seed=random_seed))
    mit.consume(map(lambda i: circuit.append(sqrt_H, [i]), range(n_qubits)))

    # Get unitary
    U = cirq.unitary(
        cirq.Circuit(
            map(
                lambda gate: cirq.MatrixGate(gate.matrix).on(*map(
                    cirq.LineQubit, map(circuit.qubits.index, gate.qubits))),
                circuit)))

    # Load TN
    arrays, ts_inds, output_inds = load(circuit,
                                        initial_state=None,
                                        final_state=None)

    # Contract TN
    tn = TensorNetwork(map(Tensor, arrays,
                           ts_inds)).contract(output_inds=output_inds)
    tn = tn.reindex_(dict(map(lambda x: (x, (x[0], int(x[1] > 0))), tn.inds)))
    tn_ = tn.fuse_(
        dict(L=tuple(zip(circuit.qubits, its.repeat(1))),
             R=tuple(zip(circuit.qubits, its.repeat(0)))))

    # Check
    np.testing.assert_allclose(U, tn_.data, atol=1e-5)
