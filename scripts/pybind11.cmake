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

include(${CMAKE_SOURCE_DIR}/scripts/tools.cmake)

find_package(pybind11)

# Try to download pybind11
if (NOT pybind11_FOUND AND DEFINED ENV{DOWNLOAD_DEPS})
  GitClone(pybind11 https://github.com/pybind/pybind11.git)
  InstallPackage(${pybind11_SOURCE_DIR} install/fast)
  find_package(pybind11 PATHS ${CMAKE_BINARY_DIR}/external/share/cmake/pybind11)
endif()

if (NOT pybind11_FOUND)
  message(FATAL_ERROR "'pybind11' not found.")
endif()
