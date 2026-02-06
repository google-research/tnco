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

import pytest


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
