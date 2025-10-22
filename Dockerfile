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

FROM debian:stable-slim AS build

# Install baseline
RUN apt -y update && \
    apt -y upgrade && \
    apt -y install cmake \
                   g++ \
                   git \
                   libmpfr-dev \
                   libboost-dev \
                   python3-dev \
                   python3-venv && \
    apt -y clean && \
    apt -y autoclean && \
    apt -y autoremove

# Install virtual environment
RUN mkdir -p /opt/tnco && \
    python3 -m venv /opt/tnco

# Copy source
COPY . /tnco/

# Install TNCO
RUN /opt/tnco/bin/pip install --no-cache-dir /tnco[parallel]

# Fresh image
FROM debian:stable-slim

# Install baseline
RUN apt -y update && \
    apt -y upgrade && \
    apt -y install python3-venv \
                   libmpfr6 && \
    apt -y clean && \
    apt -y autoclean && \
    apt -y autoremove

# Copy environment
COPY --from=build /opt/tnco /opt/tnco/

# Fix entrypoint
ENTRYPOINT ["/opt/tnco/bin/tnco"]
