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

from typing import Optional, Iterable, Tuple, List, FrozenSet, Dict
from tnco.typing import Index, Array
from string import ascii_letters
import more_itertools as mit
from random import Random
import functools as fts
import itertools as its
import operator as op
import autoray as ar

__all__ = ['decompose_hyper_inds', 'get_einsum_path', 'svd']


def is_diagonal(array: Array, /, *, atol: Optional[float] = 1e-8) -> bool:
    """Check if array 'array' is diagonal.

    Check if array 'array' is diagonal.

    Args:
        array: Array to check.
        atol: Absolute tollerance while checking non-diagonal elements.

    Returns:
        'True' if 'array' is diagonal.
    """
    # Convert to array
    array = ar.do('asarray', array)

    # Check number of dimensions
    if array.ndim <= 1:
        raise ValueError("The array must have at least two dimensions.")

    # If axes have different sizes, the array cannot be diagonal
    if array.shape[0] != array.shape[1]:
        return False

    # Remove diagonal
    array = array - ar.do('asarray', [[
        array[i_, i_] if i_ == j_ else ar.do('zeros', array.shape[2:])
        for j_ in range(array.shape[0])
    ]
                                      for i_ in range(array.shape[1])])

    # Check if all other elements are zero
    return ar.do('allclose', array, 0, atol=atol)


def decompose_hyper_inds(
    array: Array,
    inds: Iterable[Index],
    *,
    atol: Optional[float] = 1e-8,
    **kwargs
) -> Tuple[Tuple[Array, List[Index]], Dict[Index, FrozenSet[Index]]]:
    """Decompose 'array' in hyper-indices.

    Decompose 'array' in hyper-indices.

    Args:
        array: Array representing the tensor.
        inds: Indices of the tensor.
        atol: Absolute tollerance when checking for hyper-indices.

    Returns:
        It returns the reduced tensors and a map on how the indices have been
        merged.

    Raises:
        ValueError: If arguments are not consistent within each other.
    """

    # Get cache
    _hyper_inds = kwargs.pop('_hyper_inds', None)
    if kwargs:
        raise TypeError("Got unexpected keyword arguments.")

    # Convert to array
    array = ar.do('asarray', array)

    # Check number of inds
    inds = tuple(inds)
    if array.ndim != len(inds):
        raise ValueError("Wrong number of indices.")
    if next(mit.duplicates_everseen(inds), None) is not None:
        raise ValueError("'inds' has duplicated indices.")

    # Create a new dict of merged inds is not provided
    if _hyper_inds is None:
        _hyper_inds = {}

    # Pad dimensions
    def pad(xs):
        return tuple(xs) + tuple(
            filter(lambda x: x not in xs, range(array.ndim)))

    # Get first pair of hyper-inds
    h_inds = next(((i, j)
                   for i in range(array.ndim)
                   for j in range(i + 1, array.ndim)
                   if is_diagonal(array.transpose(pad((i, j))), atol=atol)),
                  None)
    if h_inds is None:
        return (array, inds), _hyper_inds

    # Transpose
    inds = tuple(map(lambda x: inds[x], pad(h_inds)))
    array = array.transpose(pad(h_inds))

    # Reduce
    _hyper_inds[inds[1]] = _hyper_inds.get(
        inds[0], frozenset()) | _hyper_inds.get(inds[1],
                                                frozenset()) | {inds[0]}
    _hyper_inds.pop(inds[0], None)
    inds = inds[1:]
    array = ar.do('stack',
                  list(map(lambda x: array[x, x], range(array.shape[0]))))

    # Call again
    return decompose_hyper_inds(array, inds, _hyper_inds=_hyper_inds)


def get_einsum_path(inds_a: Iterable[Index], inds_b: Iterable[Index],
                    output_inds: Iterable[Index], /) -> str:
    """Return einsum path.

    Return einsum path for the contraction 'inds_a @ inds_b -> output_inds'.

    Args:
        inds_a: Indices of the contracted tensor.
        inds_b: Indices of the contracted tensor.
        output_inds: Indices of the output tensor.

    Returns:
        The corresponding einsum path.
    """
    # Normalize indexes
    cntr = dict(
        zip(mit.unique_everseen(its.chain(inds_a, inds_b, output_inds)),
            ascii_letters))

    # Return path
    return ''.join(map(cntr.get, inds_a)) + ',' + ''.join(map(
        cntr.get, inds_b)) + '->' + ''.join(map(cntr.get, output_inds))


def svd(array: Array,
        inds: Iterable[Index],
        left_inds: Iterable[Index],
        *,
        svd_index_name: Optional[any] = None,
        atol: Optional[float] = 1e-8,
        seed: Optional[int] = None) -> List[Tuple[Array, Tuple[Index, ...]]]:
    """Decompose array.

    Create a new tensors by decomposing the provided one using the singular
    value decomposition.

    Args:
        array: array to decompose.
        inds: List of indices for 'array'.
        left_inds: List of indices to gather and split from the rest.
        svd_index_name: Name for the extra SVD index.
        atol: Remove all singular values smaller than 'atol'.
        seed: Seed to use when generating the new index after decomposition.

    Returns:
        The tensors obtained by decomposing 'array'.

    Raises:
        ValerError: If arguments are not consistent with each other.
    """

    # Convert
    array = ar.do('asarray', array)
    inds = tuple(inds)
    left_inds = tuple(left_inds)

    # Check
    if array.ndim != len(inds):
        raise ValueError("Wrong number of indices.")
    if not frozenset(left_inds).issubset(inds):
        raise ValueError("'left_inds' must be a subset of 'inds'.")
    if svd_index_name in inds:
        raise ValueError("'svd_index_name' must be different from 'inds'.")

    # If svd_index_name is not provided, generate a random one
    if svd_index_name is None:
        while (svd_index_name :=
               ''.join(Random(seed).choices(ascii_letters, k=10))) in inds:
            pass

    # If left_inds is empty or equal to inds, just return
    if len(left_inds) in [0, array.ndim]:
        left_inds = inds if len(left_inds) == 0 else left_inds
        return [(array.transpose(tuple(map(inds.index, left_inds))), left_inds)]

    # Get dimensions
    dims = dict(zip(inds, array.shape))

    # Get right inds
    right_inds = tuple(filter(lambda x: x not in left_inds, inds))

    # Get dimension of left inds
    left_size = fts.reduce(op.mul, map(dims.get, left_inds), 1)

    # Transpose array to have the left inds on the left, and right inds on the
    # right
    array = array.transpose(tuple(map(inds.index,
                                      left_inds + right_inds))).reshape(
                                          (left_size, -1))

    # Apply SVD
    U, s, Vh = ar.do('linalg.svd', array, full_matrices=False)

    # Get only those elements whose singular values are larger than the
    # threshold
    pos = (s >= atol)
    U = U[:, pos]
    s = s[pos]
    Vh = Vh[pos]

    # Reshape U and Vh to the right dimensions
    U = U.reshape(tuple(map(dims.get, left_inds)) + (-1,))
    Vh = Vh.reshape((-1,) + tuple(map(dims.get, right_inds)))

    # Return the new tensors
    return [U, (*left_inds, svd_index_name)
           ], [s, (svd_index_name,)], [Vh, (svd_index_name, *right_inds)]
