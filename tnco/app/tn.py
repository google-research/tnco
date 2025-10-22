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

from typing import Optional, Dict, Tuple, FrozenSet, Any, List
from tnco.typing import Index, Matrix, Array
from types import MappingProxyType
from dataclasses import dataclass
from collections import Counter
import more_itertools as mit
import itertools as its
import operator as op
import autoray as ar
import json

__all__ = ['Tensor', 'TensorNetwork']


class JSONEncoder(json.JSONEncoder):
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

        if isinstance(obj, complex):
            return '{} + {}j'.format(obj.real, obj.imag)
        if isinstance(obj, frozenset):
            return tuple(obj)
        if isinstance(obj, Tensor):
            return dict(inds=obj.inds,
                        dims=obj.dims,
                        array=None if obj.array is None else obj.array.tolist(),
                        tags=obj.tags)
        if isinstance(obj, TensorNetwork):
            return dict(tensors=obj.tensors,
                        output_inds=obj.output_inds,
                        sparse_inds=obj.sparse_inds)
        if type(obj).__module__.startswith('cirq.') and type(
                obj).__name__.endswith('Qubit'):
            return repr(obj)
        if hasattr(obj, 'to_json'):
            return obj.to_json()

        return super().default(obj)


@dataclass(frozen=True, repr=False, eq=False)
class Tensor:
    """Tensor.

    Object representing a single tensor.

    Args:
        inds: A list of indices.
        dims: Dimensions associated to each index. It must be provided if
            'array' is not provided.
        array: Matrix associated to the tensor. It must be provided if 'dims'
            is not provided.
        tags: Tags associated to the tensor.
    """
    inds: Tuple[Index]
    dims: Optional[Tuple[int]] = None
    array: Optional[Matrix] = None
    tags: Optional[Dict[any, any]] = None

    def __post_init__(self):

        def is_int(x):
            try:
                return int(x) == x
            except (ValueError, TypeError):
                return False

        # At least 'dims' or 'array' must be provided
        if self.dims is None and self.array is None:
            raise ValueError("One of 'dims' or 'array' must be provided.")

        # Convert
        object.__setattr__(self, 'inds', tuple(self.inds))
        if self.array is not None:
            object.__setattr__(self, 'array', ar.do('asarray', self.array))
        if self.dims is None:
            object.__setattr__(self, 'dims', self.array.shape)
        else:
            try:
                dims = int(self.dims)
            except (TypeError, ValueError):
                object.__setattr__(self, 'dims', tuple(self.dims))
            else:
                if dims != self.dims or dims < 1:
                    raise ValueError("'dims' must be a positive integer.")
                object.__setattr__(
                    self, 'dims', tuple(its.repeat(dims, times=len(self.inds))))
        object.__setattr__(self, 'tags',
                           {} if self.tags is None else dict(self.tags))

        # Check dimensions
        if any(not is_int(d) or d < 1 for d in self.dims):
            raise ValueError("Every dimension must be a positive integers.")
        if len(self.dims) != len(self.inds):
            raise ValueError("Wrong number of 'inds'.")

        # Check array
        if self.array is not None and self.array.shape != self.dims:
            raise ValueError("'dims' are not consistent with 'array'.")

    def __eq__(self, other, *, atol: float = 1e-5):
        if (self.array is None) ^ (other.array is None):
            return False
        if self.array is not None and not all(
                abs(self.array - other.array).ravel() < atol):
            return False
        return self.inds == other.inds and self.dims == other.dims

    def __repr__(self):
        return 'Tensor(ndim={}, array={}{}{})'.format(
            self.ndim, None if self.array is None else self.array.shape,
            '' if self.array is None else ', dtype={}'.format(self.array.dtype),
            '' if not self.tags else ', tags={}'.format(self.tags))

    @property
    def ndim(self) -> int:
        """Number of dimensions.

        Number of dimensions.

        Returns:
            The number of dimensions.
        """
        return len(self.dims)

    def to_json(self) -> Any:
        """Dump to JSON format.

        Dump the tensor in JSON format.

        Returns:
            The tensor in JSON format.
        """

        return json.dumps(self, cls=JSONEncoder)


@dataclass(frozen=True, repr=False)
class TensorNetwork:
    """Tensor network.

    Object representing the tensor network.

    Args:
        tensors: List of tensors.
        output_inds: Output indices of 'TensorNetwork'. It must be provided if
            'TensorNetwork' has hyper-indices.
        sparse_inds: List of indices to consider sparse.
        tags: Tags associated to the tensor network.
    """
    tensors: Tuple[Tensor]
    output_inds: Optional[FrozenSet[Index]] = None
    sparse_inds: Optional[FrozenSet[Index]] = None
    tags: Optional[Dict[Any, Any]] = None

    def __post_init__(self):
        # Convert
        object.__setattr__(self, 'tensors', tuple(self.tensors))
        if self.output_inds is not None:
            object.__setattr__(self, 'output_inds', frozenset(self.output_inds))
        object.__setattr__(
            self, 'sparse_inds',
            frozenset(() if self.sparse_inds is None else self.sparse_inds))

        # Check tensors
        if any(not isinstance(t, Tensor) for t in self.tensors):
            raise ValueError("'tensors' must be a list of valid 'Tensor'.")

        # Get all indices
        object.__setattr__(
            self, '_inds',
            frozenset(mit.flatten(map(lambda t: t.inds, self.tensors))))

        # Compute dimensions
        object.__setattr__(
            self, '_dims',
            dict(
                mit.flatten(
                    map(lambda t: list(mit.transpose([t.inds, t.dims])),
                        self.tensors))))

        # Check consistency of dimensions
        if any(
                map(lambda t: t.dims != tuple(map(self.dims.get, t.inds)),
                    self.tensors)):
            raise ValueError("Dimensions of 'tensors' are not consistent.")

        # Get hyper-count
        hyper_count = dict(
            its.starmap(lambda x, n: (x, n - 1),
                        Counter(mit.flatten(self.ts_inds)).items()))

        if self.output_inds is None:
            # Raise an error if there are hyper-inds but output_inds has not
            # been provided
            if any(map(lambda x: x > 1, hyper_count.values())):
                raise ValueError("'output_inds' must be provided if "
                                 "'ts_inds' has hyper-indices.")

            # Otherwise, generate output inds from hyper_count
            object.__setattr__(
                self, 'output_inds',
                map(op.itemgetter(0),
                    filter(lambda x: x[1] == 0, hyper_count.items())))

        # Convert to set
        object.__setattr__(self, 'output_inds', frozenset(self.output_inds))

        # Check output indices
        if not self.output_inds.issubset(mit.flatten(self.ts_inds)):
            raise ValueError("'output_inds' is not consistent with 'ts_inds'.")

        # Check output_inds / sparse_inds
        if not self.output_inds.issubset(self.inds):
            raise ValueError("'output_inds' contains indices not in 'tensors'.")
        if not self.sparse_inds.issubset(self.inds):
            raise ValueError("'sparse_inds' contains indices not in 'tensors'.")

        # Convert tags
        object.__setattr__(self, 'tags',
                           dict(() if self.tags is None else self.tags))

    def __repr__(self):
        return 'TensorNetwork(n_tensors={}, n_inds={})'.format(
            self.n_tensors, self.n_inds)

    @property
    def n_tensors(self) -> int:
        """Number of tensors.

        Number of tensors.

        Returns:
            The total number of tensors.
        """
        return len(self.tensors)

    @property
    def n_inds(self) -> int:
        """Number of indices.

        Number of indices.

        Returns:
            The total number of indices.
        """
        return len(self.inds)

    @property
    def ts_inds(self) -> List[List[Index]]:
        """List of indices for each tensor.

        Returns the indices associated to each tensor.

        Returns:
            A list of indices for each tensor.
        """
        return tuple(map(lambda t: t.inds, self.tensors))

    @property
    def arrays(self) -> List[Array]:
        """List of arrays for each tensor.

        Returns the array associated to each tensor.

        Returns:
            A list of indices for each tensor.
        """
        return tuple(map(lambda t: t.array, self.tensors))

    @property
    def ts_tags(self) -> List[Dict[Any, Any]]:
        """List of tags for each tensor.

        Returns the tags associated to each tensor.

        Returns:
            A list of tags for each tensor.
        """
        return tuple(map(lambda t: t.tags, self.tensors))

    @property
    def inds(self) -> FrozenSet[Index]:
        """Set of indices.

        Return all indices associated to the tensor network.

        Returns:
            All indices associated to the tensor network.
        """
        return self._inds

    @property
    def dims(self) -> Dict[Any, int]:
        """Map of dimensions.

        Return the dimensions of each index.

        Returns:
            A map of indices and their dimension.
        """
        return MappingProxyType(self._dims)

    def __len__(self):
        return self.n_tensors

    def __getitem__(self, key):
        return self.tensors[key]

    def __iter__(self):
        return iter(self.tensors)

    def to_json(self):
        return json.dumps(self, cls=JSONEncoder)
