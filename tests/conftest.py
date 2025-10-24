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

import signal
from os import environ
from random import randrange

import pytest

# Global seed
global_seed = None
fraction_n_tests = 0.1


def pytest_configure(config):
    # Randomize hash seed
    environ["PYTEST_SEED"] = environ.get("PYTEST_SEED", str(randrange(2**32)))


def pytest_sessionstart(session):
    # Assign global seed
    global global_seed
    global_seed = environ["PYTEST_SEED"]
    print(f'seed: {global_seed}')

    # Set maximum number of tests
    global fraction_n_tests
    fraction_n_tests = float(environ.get('PYTEST_FRACTION_N_TESTS', 10)) / 100
    print(f'fraction_n_tests: {fraction_n_tests * 100:1.0f}%')


@pytest.fixture
def timeout(request):

    def handler(*args):
        pytest.skip("It's taking too long, giving up.")

    # Get timeout from fixture
    if (timeout := request.node.get_closest_marker("timeout")):
        timeout = timeout.args[0]
    else:
        raise ValueError("@pytest.mark.timeout() must be used.")

    # Initialize alarm
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)

    # Run test
    try:
        yield

    # Reset alarm
    finally:
        signal.alarm(0)
