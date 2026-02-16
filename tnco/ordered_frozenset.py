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
"""Ordered Frozen Set."""

from itertools import chain
from typing import Any, Iterable, Iterator

from more_itertools import unique_everseen

__all__ = ['OrderedFrozenSet']


class OrderedFrozenSet:
    """An ordered immutable set.

    Similar to ``frozenset``, but maintains the insertion order of elements.

    Args:
        iterable: An iterable of elements to add to the set.

    Examples:
        >>> from tnco.ordered_frozenset import OrderedFrozenSet
        >>> s = OrderedFrozenSet([3, 1, 2, 1])
        >>> s
        OrderedFrozenSet((3, 1, 2))
        >>> list(s)
        [3, 1, 2]
    """

    def __init__(self, iterable: Iterable[Any] = (), /) -> None:
        self._order = tuple(unique_everseen(iterable))
        self._set = frozenset(self._order)

    def __repr__(self) -> str:
        return '{}({})'.format(type(self).__name__, self._order)

    def __str__(self) -> str:
        return repr(self)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._order)

    def __hash__(self) -> int:
        return hash(self._order)

    def __len__(self) -> int:
        return len(self._order)

    def copy(self) -> 'OrderedFrozenSet':
        """Returns a shallow copy of itself.

        Returns a shallow copy of this ``OrderedFrozenSet``.

        Returns:
            A shallow copy of itself.
        """
        return OrderedFrozenSet(self)

    def difference(self, other: Iterable) -> 'OrderedFrozenSet':
        """Difference.

        Returns the difference with ``other``.

        Args:
            other: The other set to perform the operation with.

        Returns:
            A new ``OrderedFrozenSet`` with elements in this set that are not
            in ``other``.
        """

        if not isinstance(other, OrderedFrozenSet):
            other = OrderedFrozenSet(other)
        return OrderedFrozenSet(
            filter(lambda x: x not in other._set, self._order))

    def intersection(self, other: Iterable) -> 'OrderedFrozenSet':
        """Intersection.

        Returns the intersection with ``other``.

        Args:
            other: The other set to perform the operation with.

        Returns:
            A new ``OrderedFrozenSet`` with elements common to this set and
            ``other``.
        """

        if not isinstance(other, OrderedFrozenSet):
            other = OrderedFrozenSet(other)
        return OrderedFrozenSet(filter(lambda x: x in other._set, self._order))

    def isdisjoint(self, other: Iterable) -> bool:
        """Checks for common elements.

        Checks if this ``OrderedFrozenSet`` shares no elements with ``other``.

        Args:
            other: The other set to perform the operation with.

        Returns:
            ``True`` if there are no shared elements, ``False`` otherwise.
        """

        return self._set.isdisjoint(other)

    def issubset(self, other: Iterable) -> bool:
        """Checks if subset.

        Checks if this ``OrderedFrozenSet`` is a subset of ``other``.

        Args:
            other: The other set to perform the operation with.

        Returns:
            ``True`` if this set is a subset of ``other``, ``False`` otherwise.
        """

        return self._set.issubset(other)

    def issuperset(self, other: Iterable) -> bool:
        """Checks if superset.

        Checks if this ``OrderedFrozenSet`` is a superset of ``other``.

        Args:
            other: The other set to perform the operation with.

        Returns:
            ``True`` if this set is a superset of ``other``, ``False``
            otherwise.
        """
        return self._set.issuperset(other)

    def symmetric_difference(self, other: Iterable) -> 'OrderedFrozenSet':
        """Symmetric difference.

        Returns the symmetric difference with ``other``.

        Args:
            other: The other set to perform the operation with.

        Returns:
            A new ``OrderedFrozenSet`` with elements in either this set or
            ``other`` but not both.
        """

        if not isinstance(other, OrderedFrozenSet):
            other = OrderedFrozenSet(other)
        sd = self._set.symmetric_difference(other._set)
        return OrderedFrozenSet(
            filter(lambda x: x in sd, chain(self._order, other._order)))

    def union(self, other: Iterable) -> 'OrderedFrozenSet':
        """Union.

        Returns the union with ``other``.

        Args:
            other: The other set to perform the operation with.

        Returns:
            A new ``OrderedFrozenSet`` with elements from both sets.
        """

        if not isinstance(other, OrderedFrozenSet):
            other = OrderedFrozenSet(other)
        return OrderedFrozenSet(chain(self._order, other._order))

    def __or__(self, other: Any) -> 'OrderedFrozenSet':
        if not isinstance(other, (OrderedFrozenSet, set, frozenset)):
            raise TypeError(
                "unsupported operand type(s) for |: '{}' and '{}'".format(
                    type(self).__name__,
                    type(other).__name__))
        return self.union(other)

    def __and__(self, other: Any) -> 'OrderedFrozenSet':
        if not isinstance(other, (OrderedFrozenSet, set, frozenset)):
            raise TypeError(
                "unsupported operand type(s) for &: '{}' and '{}'".format(
                    type(self).__name__,
                    type(other).__name__))
        return self.intersection(other)

    def __xor__(self, other: Any) -> 'OrderedFrozenSet':
        if not isinstance(other, (OrderedFrozenSet, set, frozenset)):
            raise TypeError(
                "unsupported operand type(s) for ^: '{}' and '{}'".format(
                    type(self).__name__,
                    type(other).__name__))
        return self.symmetric_difference(other)

    def __sub__(self, other: Any) -> 'OrderedFrozenSet':
        if not isinstance(other, (OrderedFrozenSet, set, frozenset)):
            raise TypeError(
                "unsupported operand type(s) for ^: '{}' and '{}'".format(
                    type(self).__name__,
                    type(other).__name__))
        return self.difference(other)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, OrderedFrozenSet):
            return self._set < other._set
        if isinstance(other, (set, frozenset)):
            return self._set < other
        raise TypeError(
            "unsupported operand type(s) for <: '{}' and '{}'".format(
                type(self).__name__,
                type(other).__name__))

    def __le__(self, other: Any) -> bool:
        if isinstance(other, OrderedFrozenSet):
            return self._set <= other._set
        if isinstance(other, (set, frozenset)):
            return self._set <= other
        raise TypeError(
            "unsupported operand type(s) for <=: '{}' and '{}'".format(
                type(self).__name__,
                type(other).__name__))

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, OrderedFrozenSet):
            return self._set > other._set
        if isinstance(other, (set, frozenset)):
            return self._set > other
        raise TypeError(
            "unsupported operand type(s) for >: '{}' and '{}'".format(
                type(self).__name__,
                type(other).__name__))

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, OrderedFrozenSet):
            return self._set >= other._set
        if isinstance(other, (set, frozenset)):
            return self._set >= other
        raise TypeError(
            "unsupported operand type(s) for >=: '{}' and '{}'".format(
                type(self).__name__,
                type(other).__name__))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, OrderedFrozenSet):
            return self._set == other._set
        if isinstance(other, (set, frozenset)):
            return self._set == other
        raise TypeError(
            "unsupported operand type(s) for == '{}' and '{}'".format(
                type(self).__name__,
                type(other).__name__))
