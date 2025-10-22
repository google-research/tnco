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

# Set the minimum required version for cmake
cmake_minimum_required(VERSION 3.15)

# Set the project name
set(project_name "tnco_core")
project(${project_name} LANGUAGES CXX)

# Set the C++ standard to c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Print build type
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# Add options
add_compile_options(-Wall -Wpedantic -Wfatal-errors)

# Use FindPython or FindPython3 module
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)

# Include boost library
include(${CMAKE_SOURCE_DIR}/scripts/boost.cmake)

# Include pybind11
include(${CMAKE_SOURCE_DIR}/scripts/pybind11.cmake)

# Include MPFR
include(${CMAKE_SOURCE_DIR}/scripts/mpfr.cmake)

# Include TNCO headers
include_directories(${CMAKE_SOURCE_DIR}/include/)

# Get all source files
file(GLOB_RECURSE SOURCES "./include/tnco/*.cpp")

# Initialize PyBind11 library
pybind11_add_module(${project_name} ${SOURCES})

# Install core
install(TARGETS ${project_name} DESTINATION .)
