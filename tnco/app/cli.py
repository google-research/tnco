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

from inspect import Signature, Parameter, signature
from more_itertools import value_chain
from tnco.app import Optimizer
from typing import Callable
from fire import Fire


def wraps(fn: Callable) -> Callable:
    """Apply 'fn' signature.

    'wraps' is a decorator to add 'self' + 'fn' signature, and 'fn.__doc__' to
    a function. Useful when a non-method function is added as a method.

    Args:
        fn: Function to copy the signature and the docstring from.

    Returns:
        Decorator.
    """

    def wrapper(gn):
        # Update signature
        gn.__signature__ = Signature(parameters=value_chain(
            Parameter('self', kind=Parameter.POSITIONAL_ONLY),
            signature(fn).parameters.values()))

        # Add doc
        gn.__doc__ = fn.__doc__

        # Return function
        return gn

    # Return wrapper
    return wrapper


class Main:

    @wraps(Optimizer)
    def __init__(self, *args):
        # Convert to keyword arguments
        params = dict(zip(signature(Optimizer).parameters, args))

        # Set JSON as default format if not provided
        if params['output_format'] is None:
            params['output_format'] = 'json'

        # Get optimizer
        self._optimizer = Optimizer(**params)
        self.optimize = self._optimizer.optimize


# Add documentation
Main.__doc__ = Optimizer.__doc__


def main():
    Fire(Main)
