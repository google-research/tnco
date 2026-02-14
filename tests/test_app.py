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
import json
import operator as op
import pickle
from decimal import Decimal
from os import environ
from random import Random
from tempfile import NamedTemporaryFile

import cirq
import more_itertools as mit
import numpy as np
import pytest
from cirq.contrib.qasm_import import circuit_from_qasm
from qsimcirq import QSimOptions, QSimSimulator
from quimb.tensor import Tensor, TensorNetwork

from tnco.app import Optimizer
from tnco.app import Tensor as TS
from tnco.app import TensorNetwork as TN
from tnco.app import load_tn
from tnco.app.circuit import Sampler
from tnco.testing.utils import generate_random_tensors

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
def test_LoadTN_CirqCircuit(random_seed):

    def check_tn(U, tn):
        tn = TensorNetwork(map(Tensor, tn.arrays,
                               tn.ts_inds)).contract(output_inds=tn.output_inds)
        tn = tn.transpose(*sorted(tn.inds, key=lambda x: (x[1] == 'i', x[0])))
        np.testing.assert_allclose(U.ravel(), tn.data.ravel(), atol=1e-5)

    # Get RNG
    rng = Random(random_seed)

    # Get random seed
    circuit_seed = rng.randrange(2**30)

    # How to load tn
    load_tn_ = fts.partial(load_tn,
                           initial_state=None,
                           final_state=None,
                           seed=circuit_seed)

    # Set number of qubits and depth of the circuit
    n_qubits = 8
    circuit_depth = 16

    # Get random circuit
    circuit = cirq.testing.random_circuit(n_qubits,
                                          circuit_depth,
                                          1,
                                          random_state=circuit_seed)

    # Get unitary
    U = cirq.unitary(circuit)

    # Get tn from circuit
    check_tn(U, tn_ := load_tn_(circuit))
    assert load_tn_(circuit) == tn_

    # Store to file
    with NamedTemporaryFile(mode='w+t', delete=True) as tmp:
        # With compression
        cirq.to_json_gzip(circuit, tmp.name)
        check_tn(U, tn_ := load_tn_(tmp.name))
        assert load_tn_(circuit) == tn_

        # Without compression
        cirq.to_json(circuit, tmp.name)
        check_tn(U, tn_ := load_tn_(tmp.name))
        assert load_tn_(circuit) == tn_

    # To QASM
    check_tn(cirq.unitary(circuit_from_qasm(circuit.to_qasm())),
             load_tn_(circuit.to_qasm()))


@repeat(40)
def test_OptimizeTN(random_seed, **kwargs):
    # How to convert inds
    def convert_index(x):
        if isinstance(x, (str, int, frozenset)):
            return x
        if isinstance(x, (list, tuple)):
            return frozenset(x)
        raise NotImplementedError()

    # Conver to tuple
    def to_tuple(x):
        if isinstance(x, str):
            return x
        try:
            return tuple(map(to_tuple, x))
        except (ValueError, TypeError):
            return x

    # Get rng
    rng = Random(random_seed)

    # Get numpy rng
    rng_np = np.random.default_rng(random_seed)

    # Initialize variables
    n_tensors = kwargs.get('n_tensors', rng.randint(5, 10))
    k = kwargs.get('k', rng.randint(2, 4))
    n_inds = kwargs.get('n_inds', rng.randint(10, 15))
    n_cc = kwargs.get('n_cc', rng.randint(1, 3))
    n_output_inds = kwargs.get('n_output_inds', rng.randrange(0, 5))
    randomize_names = kwargs.get('randomize_names', rng.choice([True, False]))
    max_width = kwargs.get(
        'max_width',
        None if rng.randrange(2) else int(300 * rng.random()) / 100)

    # TODO
    sparse_inds = frozenset()

    # Token for sparse / output inds
    output_index_token = ''.join(map(chr, rng.choices(range(33, 127), k=6)))
    sparse_index_token = ''.join(map(chr, rng.choices(range(33, 127), k=6)))

    # Check minimum number of indices
    if (n_inds - n_output_inds) < n_tensors + 1 - k:
        pytest.skip("Too few indices")

    load_tn_ = fts.partial(load_tn,
                           seed=rng.randrange(2**30),
                           output_index_token=output_index_token,
                           sparse_index_token=sparse_index_token,
                           decompose_hyper_inds=(n_output_inds == 0))

    # Initialize optimizer
    opt = Optimizer(seed=rng.randrange(2**30), max_width=max_width)
    assert pickle.loads(pickle.dumps(opt)) == opt

    # Get random tensors
    ts_inds, output_inds = generate_random_tensors(
        n_tensors=n_tensors,
        n_inds=n_inds,
        k=k,
        n_cc=n_cc,
        n_output_inds=n_output_inds,
        randomize_names=randomize_names,
        seed=random_seed)

    # Get random dimensions
    dims = dict(
        map(lambda x:
            (x, rng.randrange(1, 4)), mit.unique_everseen(
                mit.flatten(ts_inds)))) if rng.randrange(2) else rng.randrange(
                    1, 4)
    dims_ = dict(
        zip(mit.unique_everseen(mit.flatten(ts_inds)),
            its.repeat(dims))) if isinstance(dims, int) else dims

    # Get random arrays
    arrays = list(
        map(lambda xs: rng_np.normal(0, 1, size=tuple(map(dims_.get, xs))),
            ts_inds))

    # Get exact tn
    exact_tn = TensorNetwork(map(Tensor, arrays,
                                 ts_inds)).contract(output_inds=output_inds)

    # Load TN and contract
    tn = load_tn_(
        TN(map(lambda xs, a: TS(xs, array=a), ts_inds, arrays),
           output_inds=output_inds))
    tn = TensorNetwork(map(lambda t: Tensor(t.array, t.inds),
                           tn)).contract(output_inds=tn.output_inds)

    # Check if close
    if isinstance(tn, Tensor):
        np.testing.assert_allclose(tn.transpose_like(exact_tn).data,
                                   exact_tn.data,
                                   atol=1e-5)
    else:
        np.testing.assert_allclose(tn, exact_tn, atol=1e-5)

    # Get contraction
    tn, res = opt.optimize(TN(map(lambda xs, a: TS(xs, array=a), ts_inds,
                                  arrays),
                              output_inds=output_inds),
                           betas=(0, 100),
                           n_steps=100,
                           n_runs=4,
                           output_index_token=output_index_token,
                           sparse_index_token=sparse_index_token,
                           fuse=False,
                           decompose_hyper_inds=(n_output_inds == 0))
    assert len(res) == 4

    # Read tn from json
    tn_json = json.loads(tn.to_json())
    tn_json = TN(map(
        lambda t: TS(map(convert_index, t['inds']),
                     t['dims'],
                     array=t['array'],
                     tags=t['tags']), tn_json['tensors']),
                 output_inds=map(convert_index, tn_json['output_inds']),
                 sparse_inds=tn_json['sparse_inds'])

    # Check JSON for TensorNetwork
    assert all(
        map(
            lambda tx, ty: tx.dims == ty.dims and tuple(
                map(convert_index, tx.inds)) == ty.inds and tx.tags == ty.tags
            and np.allclose(tx.array, ty.array, atol=1e-5), tn.tensors,
            tn_json.tensors))

    # Read results from json
    res_json = list(map(json.loads, map(lambda r: r.to_json(), res)))

    # Check JSON for results
    assert all(
        map(
            lambda rx, ry: to_tuple(rx.path) == to_tuple(ry[
                'path']) and to_tuple(rx.disconnected_paths) == to_tuple(ry[
                    'disconnected_paths']) and rx.cost == Decimal(ry['cost'])
            and rx.runtime_s == ry['runtime_s'], res, res_json))
    if max_width is not None:
        assert all(
            mit.flatten(
                map(
                    lambda rx, ry:
                    (frozenset(map(convert_index, rx.slices)) == frozenset(
                        map(convert_index, ry['slices'])), *map(
                            lambda sx, sy: frozenset(map(convert_index, sx)) ==
                            frozenset(map(convert_index, sy)), rx.
                            disconnected_slices, ry['disconnected_slices'])),
                    res, res_json)))

    # There should be n_cc disconnected paths
    assert dims == 1 or all(
        map(lambda r: len(r.disconnected_paths) == n_cc, res))

    # The costs should be ordered
    assert all(
        its.starmap(lambda x, y: x.cost <= y.cost, mit.sliding_window(res, 2)))

    # If max_width is not None, slices should be present
    if max_width is None:
        assert all(map(lambda x: not hasattr(x, 'slices'), res))
    else:
        assert all(map(lambda x: hasattr(x, 'slices'), res))

    # Conctract
    tn = TensorNetwork(map(lambda t: Tensor(t.array, t.inds),
                           tn)).contract(output_inds=tn.output_inds,
                                         optimize=res[0].path)
    # Check if close
    if isinstance(tn, Tensor):
        np.testing.assert_allclose(tn.transpose_like(exact_tn).data,
                                   exact_tn.data,
                                   atol=1e-5)
    else:
        np.testing.assert_allclose(tn, exact_tn, atol=1e-5)

    # Get inds_map
    inds_map = dict(
        map(lambda x: (x, [dims_[x]]),
            mit.unique_everseen(mit.flatten(ts_inds))))
    for i, xs in enumerate(ts_inds):
        for x in xs:
            inds_map[x].append(i)
    for x in output_inds:
        inds_map[x].append(output_index_token)
    for x in sparse_inds:
        inds_map[x].append(sparse_index_token)
    inds_order, inds = map(tuple, mit.transpose(inds_map.items()))

    # Load tn
    tn = load_tn_(inds, fuse=False)

    # Check consistency of the indices
    assert all(
        map(lambda xs, ys: frozenset(xs) == frozenset(ys),
            map(lambda xs: tuple(map(lambda p: inds_order[p], xs)), tn.ts_inds),
            map(lambda tag: ts_inds[int(tag['name'])], tn.ts_tags)))

    # Load tn
    tn = load_tn_('\n'.join(map(lambda xs: ' '.join(map(str, xs)), inds)),
                  fuse=False)

    # Check consistency of the indices
    assert all(
        map(lambda xs, ys: frozenset(xs) == frozenset(ys),
            map(lambda xs: tuple(map(lambda p: inds_order[p], xs)), tn.ts_inds),
            map(lambda tag: ts_inds[int(tag['name'])], tn.ts_tags)))


@repeat(30)
def test_Sampling(random_seed):
    # Initialize random generator
    rng = Random(random_seed)

    # Get params
    simplify = rng.randrange(2)

    # Initialize simulator
    qsim = QSimSimulator(QSimOptions(max_fused_gate_size=4, cpu_threads=1))

    # Initialize circuit
    n_qubits = 5
    n_moments = 10
    circuit = cirq.testing.random_circuit(n_qubits,
                                          n_moments,
                                          1,
                                          random_state=random_seed)
    qubits = sorted(circuit.all_qubits())

    # Set number of samples
    n_samples = 300

    # Set peak
    peak = bin(rng.randrange(2**n_qubits))[2:].zfill(n_qubits)

    # Get random initial state
    initial_state = cirq.Circuit(
        (cirq.X if int(b) else cirq.I).on(q)
        for q, b in zip(qubits, peak)) + (cirq.H**0.5).on_each(qubits)

    # Build peaked circuit
    circuit = initial_state + circuit + cirq.inverse(circuit)

    # Get probability of peak
    peak_prob = abs(
        qsim.simulate(circuit, qubit_order=qubits).state_vector()[int(peak,
                                                                      2)])**2
    assert peak_prob > 0.1

    # Initialize sampler
    sampler = Sampler(seed=random_seed, n_jobs=1)

    # Get intermediate state
    state = sampler.sample(circuit,
                           simplify=simplify,
                           return_intermediate_state_only=True,
                           betas=(0, 1e3),
                           n_steps=20,
                           n_runs=2)

    # State should be pickeable
    state_ = pickle.loads(pickle.dumps(state))
    assert all(
        all(x[i] == y[i] for i in [0, 3, 4]) for x, y in zip(state, state_))
    assert all(
        all(np.allclose(x, y)
            for x, y in zip(x[2], y[2]))
        for x, y in zip(state, state_)
        if x[0] is not None)
    assert all(x[1].path == y[1].path
               for x, y in zip(state, state_)
               if x[0] is not None)

    # Sample bitstrings
    bitstrings, qubit_order = sampler.sample(state,
                                             n_samples=n_samples,
                                             qubit_order=qubits)
    assert qubit_order == tuple(qubits)

    # Get peak and probability
    sampled_peak, sampled_prob = mit.first(
        sorted(bitstrings.items(), key=op.itemgetter(1), reverse=True))
    assert sampled_peak == peak
    assert abs(sampled_prob - peak_prob) < 2 * 1 / np.sqrt(n_samples)
