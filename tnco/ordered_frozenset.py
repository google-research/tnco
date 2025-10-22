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

from more_itertools import unique_everseen
from typing import Iterable, Any
from itertools import chain

__all__ = ['OrderedFrozenSet']


class OrderedFrozenSet:
    pass


class OrderedFrozenSet:
    """Ordered 'frozenset'.

    Similar to 'frozenset', but the order of the elements is kept consistent.

    Args:
        iterable: iterable to add to the set.
    """

    def __init__(self, iterable: Iterable[Any] = (), /):
        self._order = tuple(unique_everseen(iterable))
        self._set = frozenset(self._order)

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, self._order)

    def __str__(self):
        return repr(self)

    def __iter__(self):
        return iter(self._order)

    def __hash__(self):
        return hash(self._order)

    def __len__(self):
        return len(self._order)

    def copy(self) -> OrderedFrozenSet:
        """Return a shallow copy of itself.

        Return a shallow copy of itself.

        Returns:
            A shallow copy of itself.
        """
        return OrderedFrozenSet(self)

    def difference(self, other: Iterable) -> OrderedFrozenSet:
        """Difference.

        Return the difference with 'other'.

        Args:
            other: The other set to perform the operation.

        Returns:
            The results of the operation.
        """

        if not isinstance(other, OrderedFrozenSet):
            other = OrderedFrozenSet(other)
        return OrderedFrozenSet(
            filter(lambda x: x not in other._set, self._order))

    def intersection(self, other: Iterable) -> OrderedFrozenSet:
        """Intersection.

        Return the intersection with 'other'.

        Args:
            other: The other set to perform the operation.

        Returns:
            The results of the operation.
        """

        if not isinstance(other, OrderedFrozenSet):
            other = OrderedFrozenSet(other)
        return OrderedFrozenSet(filter(lambda x: x in other._set, self._order))

    def isdisjoint(self, other: Iterable) -> bool:
        """Check for common elements.

        Check if this 'OrderedFrozenSet' shares elements with 'other'.

        Args:
            other: The other set to perform the operation.

        Returns:
            'True' if there are not shared elements.
        """

        return self._set.isdisjoint(other)

    def issubset(self, other: Iterable) -> bool:
        """Check if subset.

        Check if this 'OrderedFrozenSet' is a subset of 'other'.

        Args:
            other: The other set to perform the operation.

        Returns:
            'True' if subset of 'other'.
        """

        return self._set.issubset(other)

    def issuperset(self, other: Iterable) -> bool:
        """Check if superset.

        Check if this 'OrderedFrozenSet' is a superset of 'other'.

        Args:
            other: The other set to perform the operation.

        Returns:
            'True' if superset of 'other'.
        """
        return self._set.issuperset(other)

    def symmetric_difference(self, other: Iterable) -> OrderedFrozenSet:
        """Symmetryc difference.

        Return the symmetric difference with 'other'.

        Args:
            other: The other set to perform the operation.

        Returns:
            The results of the operation.
        """

        if not isinstance(other, OrderedFrozenSet):
            other = OrderedFrozenSet(other)
        sd = self._set.symmetric_difference(other._set)
        return OrderedFrozenSet(
            filter(lambda x: x in sd, chain(self._order, other._order)))

    def union(self, other: Iterable) -> OrderedFrozenSet:
        """Union.

        Return the union with 'other'.

        Args:
            other: The other set to perform the operation.

        Returns:
            The results of the operation.
        """

        if not isinstance(other, OrderedFrozenSet):
            other = OrderedFrozenSet(other)
        return OrderedFrozenSet(chain(self._order, other._order))

    def __or__(self, other):
        if not isinstance(other, (OrderedFrozenSet, set, frozenset)):
            raise TypeError(
                "unsupported operand type(s) for |: '{}' and '{}'".format(
                    type(self).__name__,
                    type(other).__name__))
        return self.union(other)

    def __and__(self, other):
        if not isinstance(other, (OrderedFrozenSet, set, frozenset)):
            raise TypeError(
                "unsupported operand type(s) for &: '{}' and '{}'".format(
                    type(self).__name__,
                    type(other).__name__))
        return self.intersection(other)

    def __xor__(self, other):
        if not isinstance(other, (OrderedFrozenSet, set, frozenset)):
            raise TypeError(
                "unsupported operand type(s) for ^: '{}' and '{}'".format(
                    type(self).__name__,
                    type(other).__name__))
        return self.symmetric_difference(other)

    def __sub__(self, other):
        if not isinstance(other, (OrderedFrozenSet, set, frozenset)):
            raise TypeError(
                "unsupported operand type(s) for ^: '{}' and '{}'".format(
                    type(self).__name__,
                    type(other).__name__))
        return self.difference(other)

    def __lt__(self, other):
        if isinstance(other, OrderedFrozenSet):
            return self._set < other._set
        if isinstance(other, (set, frozenset)):
            return self._set < other
        raise TypeError(
            "unsupported operand type(s) for <: '{}' and '{}'".format(
                type(self).__name__,
                type(other).__name__))

    def __le__(self, other):
        if isinstance(other, OrderedFrozenSet):
            return self._set <= other._set
        if isinstance(other, (set, frozenset)):
            return self._set <= other
        raise TypeError(
            "unsupported operand type(s) for <=: '{}' and '{}'".format(
                type(self).__name__,
                type(other).__name__))

    def __gt__(self, other):
        if isinstance(other, OrderedFrozenSet):
            return self._set > other._set
        if isinstance(other, (set, frozenset)):
            return self._set > other
        raise TypeError(
            "unsupported operand type(s) for >: '{}' and '{}'".format(
                type(self).__name__,
                type(other).__name__))

    def __ge__(self, other):
        if isinstance(other, OrderedFrozenSet):
            return self._set >= other._set
        if isinstance(other, (set, frozenset)):
            return self._set >= other
        raise TypeError(
            "unsupported operand type(s) for >=: '{}' and '{}'".format(
                type(self).__name__,
                type(other).__name__))

    def __eq__(self, other):
        if isinstance(other, OrderedFrozenSet):
            return self._set == other._set
        if isinstance(other, (set, frozenset)):
            return self._set == other
        raise TypeError(
            "unsupported operand type(s) for == '{}' and '{}'".format(
                type(self).__name__,
                type(other).__name__))
