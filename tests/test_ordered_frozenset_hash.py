# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hashing must agree with the set-based equality of OrderedFrozenSet."""

import functools
import itertools
import pickle

import pytest

from tnco.ordered_frozenset import OrderedFrozenSet


@pytest.mark.parametrize("items", [(), (7,), (3, 1, 2), ("z", "a", "z"),
                                   (1, "a", (2, 3))])
def test_hash_matches_equal_builtin_frozenset(items):
    ordered = OrderedFrozenSet(items)
    frozen = frozenset(items)
    assert ordered == frozen
    assert frozen == ordered
    assert hash(ordered) == hash(frozen)


@pytest.mark.parametrize("items", list(itertools.permutations((1, 2, 3))))
def test_permutations_are_interchangeable_dictionary_keys(items):
    stored = OrderedFrozenSet((1, 2, 3))
    alternate = OrderedFrozenSet(items)
    mapping = {stored: "first"}
    assert alternate == stored
    assert mapping[alternate] == "first"
    mapping[alternate] = "updated"
    assert len(mapping) == 1
    assert next(iter(mapping)) is stored
    assert mapping[stored] == "updated"
    assert tuple(alternate) == items


@pytest.mark.parametrize("stored_type", [OrderedFrozenSet, frozenset])
@pytest.mark.parametrize("lookup_type", [OrderedFrozenSet, frozenset])
def test_cross_type_dictionary_lookup(stored_type, lookup_type):
    mapping = {stored_type((3, 2, 1)): 42}
    assert mapping[lookup_type((1, 2, 3))] == 42
    assert mapping.pop(lookup_type((2, 1, 3))) == 42
    assert not mapping


@pytest.mark.parametrize("items", [(), (1,), (2, 1), ("b", "a")])
def test_equal_sets_are_deduplicated(items):
    values = [
        OrderedFrozenSet(items),
        OrderedFrozenSet(reversed(items)),
        frozenset(items)
    ]
    assert len(set(values)) == 1
    assert len(frozenset(values)) == 1


def test_equal_arguments_share_lru_cache_entry():
    calls = []

    @functools.lru_cache(maxsize=8)
    def evaluate(indices):
        calls.append(tuple(indices))
        return len(indices)

    assert evaluate(OrderedFrozenSet((2, 1))) == 2
    assert evaluate(OrderedFrozenSet((1, 2))) == 2
    assert evaluate(frozenset((1, 2))) == 2
    assert len(calls) == 1
    assert evaluate.cache_info().hits == 2


@pytest.mark.parametrize("protocol", [0, pickle.HIGHEST_PROTOCOL])
def test_pickle_round_trip_preserves_iteration_and_lookup(protocol):
    original = OrderedFrozenSet((3, 1, 3, 2))
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert restored == original
    assert tuple(restored) == (3, 1, 2)
    assert {original: "value"}[restored] == "value"


@pytest.mark.parametrize("operation,expected", [
    ("union", (3, 1, 2, 4)),
    ("intersection", (1, 2)),
    ("difference", (3,)),
    ("symmetric_difference", (3, 4)),
])
def test_set_operations_keep_their_iteration_order(operation, expected):
    source = OrderedFrozenSet((3, 1, 2))
    result = getattr(source, operation)((2, 4, 1))
    assert tuple(result) == expected
    assert result == frozenset(expected)
    assert tuple(source) == (3, 1, 2)


def test_copy_preserves_order_and_equality():
    original = OrderedFrozenSet((3, 1, 2, 3))
    copied = original.copy()
    assert copied is not original
    assert copied == original
    assert tuple(copied) == (3, 1, 2)
    assert hash(copied) == hash(original)


def test_unhashable_elements_still_raise():
    with pytest.raises(TypeError):
        OrderedFrozenSet([[1]])
