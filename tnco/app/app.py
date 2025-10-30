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

import bz2
import functools as fts
import gzip
import io
import itertools as its
import json
import pickle
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import more_itertools as mit
from pathvalidate import ValidationError, validate_filepath

import tnco.utils.tn as tn_utils
from tnco.app.tn import Tensor, TensorNetwork
from tnco.typing import Matrix, Qubit

__all__ = ['Optimizer']


class JSONEncoder(json.JSONEncoder):

    def default(self, obj) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, BaseContractionResults):
            return dict(cost=obj.cost, runtime_s=obj.runtime_s, path=obj.path)
        if hasattr(obj, 'to_json'):
            return obj.to_json()

        return super().default(obj)


@dataclass(repr=False, frozen=True, eq=False)
class BaseContractionResults:
    """Base class for the contraction results.

    Contraction results.

    Args:
        cost: The number of operations to perform the contration for the given
            'path'.
        runtime_s: The number of seconds it took to optimize the tensor network
            and get 'path'.
        path: A path in SSA format with an expected cost of 'cost'.
    """
    cost: float
    runtime_s: float
    path: List[Tuple[int, int]]

    def __lt__(self, other):
        if not isinstance(other, BaseContractionResults):
            raise ValueError("Cannot compare against '{}'.".format(
                type(other).__name__))
        return self.cost < other.cost

    def __repr__(self):
        return 'ContractionResults(cost={:1.3g}, runtime={:1.3g}s)'.format(
            self.cost, self.runtime_s)

    def to_json(self):
        return json.dumps(self, cls=JSONEncoder)


def load_file(filename: str) -> Any:
    """Load an object from 'filename'.

    Load an object from 'filename'. The object can be of any type and the file
    can be compressed.

    Args:
        filename: Path to the file to load.

    Returns:
        Loaded object.

    Raises:
        FileNotFoundError: 'filename' does not exist or not accessible.
    """
    # Validate filename
    try:
        validate_filepath(filename, platform='auto')
    except ValidationError as e:
        raise ValueError("'filename' is not valid ({})".format(e))

    # Check that file exists
    if not Path(filename).exists:
        raise FileNotFoundError("'{}' does not exist.".format(filename))

    def load(binary):
        """
        Process binary recursively.
        """

        # Is gzip?
        if binary[:2] == b'\x1f\x8b':
            return load(gzip.decompress(binary))

        # Is bzip?
        if binary[:2] == b'BZ':
            return load(bz2.decompress(binary))

        # Is json?
        try:
            return json.loads(binary.decode())
        except json.JSONDecodeError:
            pass

        # Is text?
        try:
            return binary.decode('utf-8')
        except UnicodeDecodeError:
            pass

        # If everything fails, return it as binary
        return binary

    # Load as binary
    with open(filename, 'rb') as file:
        return load(file.read())


def load_tn(obj: Any,
            *,
            fuse: Optional[float] = 4,
            decompose_hyper_inds: Optional[bool] = True,
            simplify_circuit: Optional[bool] = True,
            initial_state: Optional[Union[str, Dict[Qubit, Matrix],
                                          None]] = '0',
            final_state: Optional[Union[str, Dict[Qubit, Matrix], None]] = '0',
            output_index_token: Optional[str] = '*',
            sparse_index_token: Optional[str] = '/',
            atol: Optional[float] = 1e-5,
            backend: Optional[str] = None,
            seed: Optional[int] = None,
            verbose: Optional[int] = False) -> TensorNetwork:
    """Load tensor network from any 'obj'.

    The function loads a tensor network from 'obj' of any type. See Notes for
    the supported formats.

    Args:
        obj: Object to load. See Notes for all possible inputs.
        fuse: Maximum width to use to fuse the tensors. The width is defined as
            sum of the logarithms of all the dimensions of a given tensor.
            Tensors are contracted so that the width of the contracted tensor
            is smaller than 'fuse'.
        decompose_hyper_inds: If 'True', decompose diagonal tensors to
            hyper-indices.
        simplify_circuit: If 'True' and 'obj' is a circuit, gates that cancel
            each other will be simplified.
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
        output_index_token: If 'obj' is a list of indices, the token to use to
            identify output inds.
        sparse_index_token: If 'obj' is a list of indices, the token to use to
            identify sparse inds.
        atol: Absolute tollerance when checking for hyper-indices.
        backend: Backend to use to fuse arrays. See: `autoray.do`.
        seed: Seed to use.
        verbose: Verbose output.

    Returns:
        The loaded tensor network.

    Raises:
        TypeError: If 'obj' is not recognized.

    Notes:
        'load_tn' accepts 'objs' in multiple formats:

        - Objects of 'tnco.app.TensorNetwork'
        - List of gates of the form:
            [
                (matrix_1, (qubit_1_1, qubits_1_2, ..)),
                (matrix_2, (qubit_2_1, qubits_2_2, ..)),
                ...
                (matrix_n, (qubit_n_1, qubits_n_2, ..)),
            ]
        - List of indices of the form:
            [
                (index_dimension_1, tensorname_1_1, tensorname_1_2, ...),
                (index_dimension_2, tensorname_2_1, tensorname_2_2, ...),
                ...
                (index_dimension_n, tensorname_n_1, tensorname_n_2, ...),
            ]
          If an index contains 'output_index_token', that index is considered
          an output index. Similarly, if an index contains
          'sparse_index_token', that index is considered a sparse index.
        - String of list of indices of the form:
          '''
          index_dimension_1 tensorname_1_1 tensorname_1_2 ...
          index_dimension_2 tensorname_2_1 tensorname_2_2 ...
          ...
          index_dimension_n tensorname_n_1 tensorname_n_2 ...
          '''
          If an index contains 'output_index_token', that index is considered
          an output index. Similarly, if an index contains
          'sparse_index_token', that index is considered a sparse index.
        - Third-parties quantum circuits, such as 'cirq.Circuit' or
          'qiskit.QuantumCircuit'
        - Third-parties quantum circuits in JSON format
        - Strings in QASM format
        - Filename where any of the above abjects is stored. The file can be
          compressed.
        - If 'stdin', read string from stdin and parse it against the above
          options.

        If 'obj' is not recognized, a 'TypeError' will be raised.
    """
    # Load all options
    options = dict(locals())

    # Remove unused
    options.pop('obj')

    # Check if object is an iterator
    def is_iterator(x):
        return isinstance(obj, Iterator)

    # Check if object is an int
    def is_int(x):
        try:
            return int(x) == x
        except (ValueError, TypeError):
            return False

    # Check if hashable
    def is_hashable(x):
        try:
            hash(x)
            return True
        except TypeError:
            return False

    # Check if random access
    def is_random_access(x):
        if is_iterator(x):
            return False
        try:
            len(x)
            x[0]
            return True
        except TypeError:
            return False

    # Is it an iterator?
    if is_iterator(obj):
        raise NotImplementedError("iterators are not supported.")

    # Check if object is a valid array
    def is_array(x):
        return is_random_access(x) and hasattr(x, 'shape') and hasattr(
            x, 'ndim')

    # Check if object is a valid matrix
    def is_matrix(x):
        return is_array(x) and x.ndim == 2 and x.shape[0] == x.shape[1]

    # Check if object is a valid gate
    def is_gate(x):
        return is_random_access(x) and len(x) == 2 and is_matrix(
            x[0]) and is_random_access(x[1]) and 2**len(x[1]) == x[0].shape[0]

    # Is it a TensorNetwork
    if isinstance(obj, TensorNetwork):
        # Get tensors
        ts_inds = list(obj.ts_inds)
        dims = obj.dims
        arrays = list(obj.arrays)
        tags = dict(obj.tags)
        ts_tags = list(obj.ts_tags)
        output_inds = obj.output_inds
        sparse_inds = obj.sparse_inds

        # Get number of provided arrays
        n_provided_arrays = sum(map(lambda a: a is not None, arrays))

        # TODO: For now, let's disable the decomposition of hyper-indices and
        # fusing gates when there are sparse indices
        if sparse_inds:
            warn("The decomposition of hyper-indices and the fusion of "
                 "indices is not yet supported if there are sparse indices")
            decompose_hyper_inds = False
            fuse = False

        # Disable hyper-indices decomposition if not all the arrays have been
        # provided
        if n_provided_arrays < len(arrays):
            if decompose_hyper_inds:
                warn("Cannot decompose hyper-indices if not "
                     "all arrays are provided.")
                decompose_hyper_inds = False

        # Disable fusing indices if only a portion of the arrays have been
        # provided
        if n_provided_arrays not in [0, len(arrays)]:
            fuse = False

        # If arrays are provided, try to decompose hyper-indices
        if decompose_hyper_inds:
            # If there are sparse indices, tn cannot be decomposed
            if sparse_inds:
                raise ValueError("Tensor network cannot be decomposed when "
                                 "there are sparse indices.")

            # Decompose hyper-indices
            arrays, ts_inds, hyper_inds_map = tn_utils.decompose_hyper_inds(
                arrays, ts_inds, atol=atol)
            output_inds = frozenset(map(hyper_inds_map.get, output_inds))

            # TODO: once tensors are decomposed, it is hard to keep track of
            # the ts_tags. For now, let's reset them
            ts_tags = [None] * len(arrays)

            # Add the map of hyper-indices to tags
            if 'hyper_inds_map' in tags:
                raise ValueError(
                    "'TensorNetwork' has already the tag 'hyper_inds_map'.")
            tags['hyper_inds_map'] = hyper_inds_map

        # Fuse if needed
        if fuse is not None and fuse > 0:
            # Get path to fuse inds
            path = tn_utils.fuse(ts_inds,
                                 dims,
                                 max_width=fuse,
                                 output_inds=output_inds,
                                 seed=seed,
                                 verbose=verbose)

            # Fuse tensors
            ts_inds, output_inds, *arrays_ = tn_utils.contract(
                path,
                ts_inds,
                output_inds,
                arrays=arrays if n_provided_arrays else None,
                dims=dims,
                backend=backend,
                verbose=(verbose - 1))
            if n_provided_arrays:
                arrays = arrays_[0]

            # Fuse ts_tags
            for (px_, py_) in map(sorted, path):
                # Get tags
                tags_y_ = ts_tags.pop(py_)
                tags_x_ = ts_tags.pop(px_)

                # Update tags
                if tags_x_ is None and tags_y_ is None:
                    ts_tags.append(None)
                elif tags_x_ is None:
                    ts_tags.append(tags_y_)
                elif tags_y_ is None:
                    ts_tags.append(tags_x_)
                else:
                    ts_tags.append(dict(x=tags_x_, y=tags_y_))

            # Update fuse paths
            if 'fuse_path' in tags:
                raise ValueError(
                    "'TensorNetwork' has already the tag 'fuse_path'.")
            tags['fuse_path'] = path

        # Return tensor network
        return TensorNetwork(map(
            lambda xs, a, tags: Tensor(
                xs, dims=map(dims.get, xs), array=a, tags=tags), ts_inds,
            arrays, ts_tags),
                             output_inds=output_inds,
                             sparse_inds=sparse_inds,
                             tags=tags)

    # Is it string?
    if isinstance(obj, str):
        # Is stdin?
        if obj == 'stdin':
            return load_tn(sys.stdin.read().strip(), **options)

        # Is it a QASM circuit? Let's load 'cirq' only if needed.
        if mit.first(
                filter(lambda x: len(x) and not x.startswith('//'),
                       obj.splitlines())).upper().startswith('OPENQASM'):
            from cirq.contrib.qasm_import import circuit_from_qasm
            return load_tn(circuit_from_qasm(obj), **options)

        # Is it a list of indices?
        if not any(
                filter(
                    fts.partial(re.match,
                                r'^(?=\s*\S)(?!#)(?!\d+(\s+\S+)*\s*$).*'),
                    obj.splitlines())):
            # Filter out comments and get map of tensors
            obj = list(
                its.starmap(
                    lambda d, *x: (int(d), *x),
                    map(
                        lambda x: re.sub(r'\s+', ' ', x).strip().split(),
                        filter(fts.partial(re.match, r'\d+(\s+\S+)*\s*$'),
                               obj.splitlines()))))

            return load_tn(obj, **options)

        # Is it a file?
        try:
            validate_filepath(obj, platform='auto')
            if Path(obj).exists():
                return load_tn(load_file(obj), **options)
        except ValidationError:
            pass

    # Is it JSON?
    if isinstance(obj, dict):

        # Is it a cirq.Circuit? Let's load 'cirq' only if needed.
        if 'cirq_type' in obj:
            from cirq import read_json
            return load_tn(read_json(io.StringIO(json.dumps(obj))), **options)

    # Is it a list of indices?
    if all(
            map(lambda x: is_random_access(x) and len(x) > 1 and is_int(x[0]),
                obj)):
        tensor_map, dims, output_inds, sparse_inds = tn_utils.read_inds(
            dict(enumerate(obj)),
            output_index_token=output_index_token,
            sparse_index_token=sparse_index_token)

        return load_tn(
            TensorNetwork(its.starmap(
                lambda name, xs: Tensor(
                    xs, map(dims.get, xs), tags=dict(name=name)),
                tensor_map.items()),
                          output_inds=output_inds,
                          sparse_inds=sparse_inds), **options)

    # Is it a list of gates?
    if all(map(is_gate, obj)):
        from tnco.utils.circuit import load

        # Convert circuit to tn
        arrays, ts_inds, output_inds = load(obj,
                                            initial_state=initial_state,
                                            final_state=final_state,
                                            simplify=simplify_circuit,
                                            decompose_hyper_inds=False,
                                            fuse=False,
                                            atol=atol,
                                            seed=seed,
                                            verbose=verbose)

        # Get tensor network
        tn = TensorNetwork(map(lambda xs, a: Tensor(xs, array=a), ts_inds,
                               arrays),
                           output_inds=output_inds)

        # Call back again
        return load_tn(tn, **options)

    # Is it a cirq.Circuit? Let's load 'cirq' only if needed.
    if type(obj).__module__.startswith('cirq.') and type(
            obj).__name__ == 'Circuit':
        from cirq import Circuit

        from tnco.utils.circuit import load

        if isinstance(obj, Circuit):
            # Convert circuit to tn
            arrays, ts_inds, output_inds = load(obj,
                                                initial_state=initial_state,
                                                final_state=final_state,
                                                simplify=simplify_circuit,
                                                decompose_hyper_inds=False,
                                                fuse=False,
                                                atol=atol,
                                                seed=seed,
                                                verbose=verbose)

            # Get tensor network
            tn = TensorNetwork(map(lambda xs, a: Tensor(xs, array=a), ts_inds,
                                   arrays),
                               output_inds=output_inds)

            # Call back again
            return load_tn(tn, **options)

    # Is it a qiskit.QuantumCircuit? Let's load 'qiskit' only if needed.
    if type(obj).__module__.startswith('qiskit.') and type(
            obj).__name__ == 'QuantumCircuit':
        from qiskit import QuantumCircuit

        from tnco.utils.circuit import load

        if isinstance(obj, QuantumCircuit):
            # Convert circuit to tn
            arrays, ts_inds, output_inds = load(obj,
                                                initial_state=initial_state,
                                                final_state=final_state,
                                                simplify=simplify_circuit,
                                                decompose_hyper_inds=False,
                                                fuse=False,
                                                atol=atol,
                                                seed=seed,
                                                verbose=verbose)

            # Get tensor network
            tn = TensorNetwork(map(lambda xs, a: Tensor(xs, array=a), ts_inds,
                                   arrays),
                               output_inds=output_inds)

            # Call back again
            return load_tn(tn, **options)

    # Not a valid object
    raise TypeError("'obj' is not recognized.")


def dump_results(tn: TensorNetwork,
                 res: List[BaseContractionResults],
                 *,
                 output_format: Optional[str] = None,
                 output_filename: Optional[str] = None,
                 output_compression: Optional[str] = 'auto',
                 overwrite_output_file: Optional[bool] = False,
                 **kwargs) -> Any:
    """Dump 'tn' and 'res' in the desired format.

    Dump 'tn' and 'res' in the desired format.

    Args:
        tn: Tensor Network to dump.
        res: Results to dump.
        output_format: Format to use for the output. See Notes for more
            details.
        output_filename: If provided, dump the output to 'output_filename'. If
            'output_filfename' has a '.gzip' or '.bz2' extension, it will be
            compressed accordingly.
        output_compression: If 'auto', the output will be compressed to the
            format specified by 'output_compression'. Otherwise, the
            compression format will be deduced from 'output_filename'. Valid
            'output_compression' are 'none', 'bz2' and 'json'. If
            'output_filename' is not provided, it will raise a 'ValueError'.
        overwrite_output_file:If 'True', the 'output_filename' will be
            overwritten if it exists.

    Returns:
        See Notes.

    Raises:
        ValueError: 'output_format' is not supported.
        ValueError: 'output_compression' is not supported or 'output_filename'
            is not provided when compressing output.
        FileExistsError: If 'output_filename' already exists, unless
            'overwrite_output_file=True'.

    Notes:
        The output format depends on the provided arguments.

        If 'output_format' is None or 'raw', the output will be the tuple '(tn,
        res)', with 'tn' being the tensor network used in the simulation and
        'res' the results from the optimization. If 'output_format' is 'json',
        both 'tn' and 'res' are dumped to the json format.

        If 'output_filename' is not specified, the function will return the
        output in the specified format. Otherwise, the output is dumped to
        'output_filename'. In this case, if 'output_format=raw', then 'pickle'
        is used to dump the results on the file.
    """
    # Check if only check is needed
    check_only = kwargs.pop('check_only', False)

    # No extra arguments
    if kwargs:
        raise TypeError("Unexpected extra keyword arguments.")

    # Check if output_format is valid
    output_format = 'raw' if output_format is None else str(
        output_format).lower()
    if output_format not in ['raw', 'json']:
        raise ValueError(f'"{output_format=}" not supported.')

    # Validate filename
    if output_filename:
        try:
            validate_filepath(output_filename, platform='auto')
        except ValidationError as e:
            raise ValueError("'output_filename' is not valid ({})".format(e))

    # Check if filename already exists
    output_filename = None if output_filename is None else Path(output_filename)
    if output_filename:
        if not overwrite_output_file and output_filename.exists():
            raise FileExistsError(
                "'{}' already exists. Please use 'overwrite_output_file=True'.".
                format(output_filename))

    # Check compression
    output_compression = str(output_compression).lower()
    if output_compression not in ['auto', 'none', 'bz2', 'gzip']:
        raise ValueError(f'"{output_compression=}" not supported.')
    if output_compression not in ['auto', 'none'] and not output_compression:
        raise ValueError(
            "Output can be compressed only if 'output_filename' is provided.")

    # Just return if check_only
    if check_only:
        return

    # Initialize output
    output = (tn, res)

    # Convert to the right format
    if output_format == 'json':
        output = '{{"tn" : {}, "res" : {}}}'.format(
            tn.to_json(),
            '[' + ', '.join(map(lambda r: r.to_json(), res)) + ']')

    # Dump to file if needed
    if output_filename:
        # Is suffix a valid compression?
        if (suffix := (output_filename.suffix[1:] if output_compression
                       == 'auto' else output_compression)) == 'gzip':
            open = gzip.open
            compress_ = True
        elif suffix == 'bz2':
            open = bz2.open
            compress_ = True
        else:
            open = io.open
            compress_ = False

        # Is the output a string?
        if isinstance(output, str):
            # Encode if output is compressed
            if compress_:
                output = output.encode()

            # Dump
            with open(output_filename, 'w') as file_:
                file_.write(output)

            # Return
            return None

        # Otherwise, dump it as pickle
        with open(output_filename, 'w' if compress_ else 'bw') as file_:
            pickle.dump(output, file_)

        # Return
        return None

    # Return from function
    return output


@dataclass(frozen=True)
class BaseOptimizer:
    """Base class for the optimizer.

    Optimize the tensor network.

    Args:
        max_width: Maximum width to use. The width is defined as sum of the
            logarithms of all the dimensions of a given tensor.  Tensors are
            contracted so that the width of the contracted tensor is smaller
            than 'max_width'.
        n_jobs: Number of processes to use. By default, all available cores are
            used. If 'n_jobs' is a positive number, 'n_jobs' processes will be
            used. If 'n_jobs' is negative, 'n_cpus + n_jobs + 1' will be used.
            If 'n_jobs' is zero, it will raise a 'ValueError'. (See:
            'tnco.parallel.Parallel')
        width_type: The type to use to represent the width. (See:
            'tnco.optimize.finite_width.cost_model.SimpleCostModel')
        cost_type: The type to use to represent the cost. (See:
            'tnco.optimize.finite_width.cost_model.SimpleCostModel')
        output_format: Format to use for the output. See Notes for more
            details.
        output_filename: If provided, dump the output to 'output_filename'. If
            'output_filfename' has a '.gzip' or '.bz2' extension, it will be
            compressed accordingly.
        output_compression: If 'auto', the output will be compressed to the
            format specified by 'output_compression'. Otherwise, the
            compression format will be deduced from 'output_filename'. Valid
            'output_compression' are 'none', 'bz2' and 'json'. If
            'output_filename' is not provided, it will raise a 'ValueError'.
        overwrite_output_file: If 'True', the 'output_filename' will be
            overwritten if it exists.
        atol: Absolute tollerance when checking for hyper-indices.
        backend: Backend to use to fuse arrays. See: `autoray.do`.
        seed: Seed to use.
        verbose: Verbose output.
    """
    max_width: Optional[float] = None
    n_jobs: Optional[int] = -1
    width_type: Optional[str] = 'float32'
    cost_type: Optional[str] = 'float64'
    output_format: Optional[str] = None
    output_filename: Optional[str] = None
    output_compression: Optional[str] = 'auto'
    overwrite_output_file: Optional[bool] = False
    atol: Optional[float] = 1e-5
    backend: Optional[str] = None
    seed: Optional[int] = None
    verbose: Optional[int] = False

    def optimize(self, *args, **kwargs):
        raise NotImplementedError()

    def _load_tn(self, tn, **load_tn_options):
        return load_tn(tn,
                       atol=self.atol,
                       backend=self.backend,
                       seed=self.seed,
                       verbose=self.verbose,
                       **load_tn_options)

    def _dump_results(self, tn, res, **dump_results_options):
        return dump_results(tn,
                            res,
                            output_format=self.output_format,
                            output_filename=self.output_filename,
                            output_compression=self.output_compression,
                            overwrite_output_file=self.overwrite_output_file,
                            **dump_results_options)

    def __post_init__(self):
        # Check dumper
        self._dump_results(None, None, check_only=True)


def Optimizer(method: Optional[str] = 'sa',
              max_width: Optional[float] = None,
              n_jobs: Optional[int] = -1,
              width_type: Optional[str] = 'float32',
              cost_type: Optional[str] = 'float64',
              output_format: Optional[str] = None,
              output_filename: Optional[str] = None,
              output_compression: Optional[str] = 'auto',
              overwrite_output_file: Optional[bool] = False,
              atol: Optional[float] = 1e-5,
              backend: Optional[str] = None,
              seed: Optional[int] = None,
              verbose: Optional[int] = False) -> BaseOptimizer:
    """Optimize the tensor network.

    Optimize the tensor network.

    Args:
        method: The method to use for the optimization.
        max_width: Maximum width to use. The width is defined as sum of the
            logarithms of all the dimensions of a given tensor.  Tensors are
            contracted so that the width of the contracted tensor is smaller
            than 'max_width'.
        n_jobs: Number of processes to use. By default, all available cores are
            used. If 'n_jobs' is a positive number, 'n_jobs' processes will be
            used. If 'n_jobs' is negative, 'n_cpus + n_jobs + 1' will be used.
            If 'n_jobs' is zero, it will raise a 'ValueError'. (See:
            'tnco.parallel.Parallel')
        width_type: The type to use to represent the width. (See:
            'tnco.optimize.finite_width.cost_model.SimpleCostModel')
        cost_type: The type to use to represent the cost. (See:
            'tnco.optimize.finite_width.cost_model.SimpleCostModel')
        output_format: Format to use for the output. See Notes for more
            details.
        output_filename: If provided, dump the output to 'output_filename'. If
            'output_filfename' has a '.gzip' or '.bz2' extension, it will be
            compressed accordingly.
        output_compression: If 'auto', the output will be compressed to the
            format specified by 'output_compression'. Otherwise, the
            compression format will be deduced from 'output_filename'. Valid
            'output_compression' are 'none', 'bz2' and 'json'. If
            'output_filename' is not provided, it will raise a 'ValueError'.
        overwrite_output_file: If 'True', the 'output_filename' will be
            overwritten if it exists.
        atol: Absolute tollerance when checking for hyper-indices.
        backend: Backend to use to fuse arrays. See: `autoray.do`.
        seed: Seed to use.
        verbose: Verbose output.

    Returns:
        The optimizer.
    """
    # Get all options
    opts = locals()

    # Remove unused
    opts.pop('method')

    # Initialize with base location
    module = 'tnco.app'

    # Infinite memory or finite witdh?
    if max_width is not None and max_width < float('inf'):
        module += '.finite_width'
    else:
        module += '.infinite_memory'

    # Add method
    module += '.' + str(method)

    # Try to import module
    module = import_module(module)

    return module.Optimizer(**opts)
