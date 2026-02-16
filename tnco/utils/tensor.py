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
"""Tensor utilities."""

import functools as fts
import itertools as its
import operator as op
from random import Random
from string import ascii_letters
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple, Union

import autoray as ar
import more_itertools as mit

from tnco.ordered_frozenset import OrderedFrozenSet
from tnco.typing import Array, Index

__all__ = ['decompose_hyper_inds', 'get_einsum_subscripts', 'tensordot', 'svd']


def is_diagonal(array: Array, /, *, atol: Optional[float] = 1e-8) -> bool:
    """Checks if an array is diagonal.

    Checks if the given array is diagonal.

    Args:
        array: Array to check.
        atol: Absolute tolerance for non-diagonal elements.

    Returns:
        ``True`` if ``array`` is diagonal, ``False`` otherwise.
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
    """Decomposes an array into hyper-indices.

    Decomposes an array into hyper-indices (diagonal structure).

    Args:
        array: Array representing the tensor.
        inds: Indices of the tensor.
        atol: Absolute tolerance for checking diagonal elements.

    Returns:
        A tuple containing the reduced tensor and a dictionary mapping new
        indices to the set of merged indices.

    Raises:
        ValueError: If arguments are inconsistent.
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

    # In permutations, all elements in 'array' are the same
    if array.size and ar.do('allclose', array, array.ravel()[0], atol=atol):
        return (array.ravel()[0] * ar.do('ones_like', array, shape=()),
                ()), _hyper_inds

    # Call again
    return decompose_hyper_inds(array, inds, _hyper_inds=_hyper_inds)


def get_einsum_subscripts(inds_a: Iterable[Index], inds_b: Iterable[Index],
                          output_inds: Iterable[Index], /) -> str:
    """Generates einsum subscripts.

    Generates the einsum subscripts string for the contraction
    ``inds_a, inds_b -> output_inds``.

    Args:
        inds_a: Indices of the first tensor.
        inds_b: Indices of the second tensor.
        output_inds: Indices of the output tensor.

    Returns:
        The corresponding einsum subscripts string.

    Examples:
        >>> from tnco.utils.tensor import get_einsum_subscripts
        >>> get_einsum_subscripts(['i', 'j'], ['j', 'k'], ['i', 'k'])
        'ab,bc->ac'
    """
    # Normalize indexes
    cntr = dict(
        zip(mit.unique_everseen(its.chain(inds_a, inds_b, output_inds)),
            ascii_letters))

    # Return path
    return ''.join(map(cntr.get, inds_a)) + ',' + ''.join(map(
        cntr.get, inds_b)) + '->' + ''.join(map(cntr.get, output_inds))


def tensordot(
    x: Tuple[Array, Iterable[Index]],
    y: Tuple[Array, Iterable[Index]],
    /,
    *,
    hyper_inds: Optional[Iterable[Index]] = None,
    return_inds_only: Optional[bool] = False
) -> Union[Tuple[Array, List[Index]], List[Index]]:
    """Contracts two tensors.

    Contracts tensor ``x`` with tensor ``y``.

    Args:
        x: First tensor (array, indices).
        y: Second tensor (array, indices).
        hyper_inds: Indices to be treated as hyper-indices (diagonal).
        return_inds_only: If ``True``, only returns the indices of the
            resulting tensor.

    Returns:
        The resulting tensor (array, indices) or just the indices if
        ``return_inds_only`` is ``True``.

    Raises:
        ValueError: If ``hyper_inds`` contains indices not shared by both
            tensors.

    Examples:
        >>> import numpy as np
        >>> from tnco.utils.tensor import tensordot
        >>> x = (np.eye(2), ['i', 'j'])
        >>> y = (np.ones(2), ['j'])
        >>> z, z_inds = tensordot(x, y)
        >>> z
        array([1., 1.])
        >>> z_inds
        ('i',)
    """
    # Get indices and arrays
    ax, ay = map(fts.partial(ar.do, 'asarray'), (x[0], y[0]))
    xs, ys = map(OrderedFrozenSet, (x[1], y[1]))

    # Get all dimensions
    dims = dict(its.chain(zip(xs, ax.shape), zip(ys, ay.shape)))

    # Check hyper-indices
    if hyper_inds is None:
        hyper_inds = ()
    hyper_inds = OrderedFrozenSet(hyper_inds)
    if not (xs & ys).issuperset(hyper_inds):
        raise ValueError("'hyper_inds' must be a list of shared indices.")

    # Let's get the shared inds
    shared_inds = (xs & ys)
    shared_no_hyper_inds = shared_inds - hyper_inds

    # Build the new x / y inds
    xs_not_shared = xs - shared_inds
    ys_not_shared = ys - shared_inds
    new_xs = tuple(hyper_inds | xs_not_shared | shared_no_hyper_inds)
    new_ys = tuple(hyper_inds | shared_no_hyper_inds | ys_not_shared)

    # Get new inds
    zs = hyper_inds | xs_not_shared | ys_not_shared

    # Return inds only if needed
    if return_inds_only:
        return tuple(zs)

    # Transpose arrays accordingly
    xs, ys = map(tuple, (xs, ys))
    ax = ax.transpose(list(map(xs.index, new_xs))).reshape(
        tuple(
            fts.reduce(op.mul, map(dims.get, xs), 1)
            for xs in (hyper_inds, xs_not_shared, shared_no_hyper_inds)))
    ay = ay.transpose(list(map(ys.index, new_ys))).reshape(
        tuple(
            fts.reduce(op.mul, map(dims.get, ys), 1)
            for ys in (hyper_inds, shared_no_hyper_inds, ys_not_shared)))

    # Return new tensor
    return (ax @ ay).reshape(tuple(map(dims.get, zs))), tuple(zs)


def svd(array: Array,
        inds: Iterable[Index],
        left_inds: Iterable[Index],
        *,
        svd_index_name: Optional[Any] = None,
        atol: Optional[float] = 1e-8,
        seed: Optional[int] = None) -> List[Tuple[Array, Tuple[Index, ...]]]:
    """Performs Singular Value Decomposition (SVD).

    Decomposes an array into three tensors using SVD: U, s, Vh.

    Args:
        array: Array to decompose.
        inds: Indices of the array.
        left_inds: Indices to keep on the U tensor.
        svd_index_name: Name for the new index created by SVD. If ``None``,
            a random name is generated.
        atol: Threshold for truncating singular values.
        seed: Seed for generating the random index name.

    Returns:
        A list containing the three tensors: [(U, U_inds), (s, s_inds), (Vh,
        Vh_inds)].

    Raises:
        ValueError: If arguments are inconsistent.

    Examples:
        >>> import numpy as np
        >>> from tnco.utils.tensor import svd
        >>> array = np.array([[1., 0.], [0., 1.]])
        >>> inds = ['i', 'j']
        >>> left_inds = ['i']
        >>> U, s, Vh = svd(array, inds, left_inds, svd_index_name='k')
        >>> U
        (array([[1., 0.],
               [0., 1.]]), ('i', 'k'))
        >>> s
        (array([1., 1.]), ('k',))
        >>> Vh
        (array([[1., 0.],
               [0., 1.]]), ('k', 'j'))
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
    return (U, (*left_inds,
                svd_index_name)), (s, (svd_index_name,)), (Vh, (svd_index_name,
                                                                *right_inds))
