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
"""Bitset implementation."""

from typing import Iterable, Optional, Union

import more_itertools as mit
from tnco_core import Bitset as Bitset_

__all__ = ['Bitset']


class Bitset(Bitset_):
    """
    Compact representation of bits.

    Args:
        bits: Either a string representing the bits or a list of positions of
            the set bits. In the latter case, the number of total bits ``n``
            must be provided.
        n: The total number of bits.

    Raises:
        ValueError: If ``bits`` are not in the right format, or too many / too
            few ``bits`` are provided.

    Examples:
        >>> from tnco.bitset import Bitset
        >>> b = Bitset([0, 2], n=4)
        >>> str(b)
        '1010'
    """

    def __init__(self,
                 bits: Optional[Union[str, Iterable[int]]] = None,
                 n: Optional[int] = None) -> None:
        if bits is None:
            # If 'bits' is not provided, 'n' must be None or 0
            if n is not None and n != 0:
                raise ValueError("'bits' must be provided.")

            # Get empty Bitset
            super().__init__()

        else:
            if isinstance(bits, str):
                # Bits can only be 0 or 1
                if any(map(lambda x: x not in '01', bits)):
                    raise ValueError("'bits' can only have '0' or '1'.")

                # Check 'n' if provided
                if n is not None and n != len(bits):
                    raise ValueError("'n' is not consistent with 'bits'.")

                # Get Bitset from string
                super().__init__(bits)

            else:
                # Convert to list
                bits = list(bits)

                # Check for repeated positions
                if next(mit.duplicates_everseen(bits), None) is not None:
                    raise ValueError("'bits' cannot have repeated positions.")

                # Check n
                if n is None:
                    raise ValueError("'n' must be provided.")
                if n < 0:
                    raise ValueError("'n' must be a non-negative integer.")

                # Check size of bits
                if len(bits) and n < max(bits):
                    raise ValueError("'n' is too small.")

                # Get Bitset from list of positions
                super().__init__(bits, n)
