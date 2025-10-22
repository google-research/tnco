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

# Find the MPFR library
find_path(MPFR_INCLUDE_DIR mpfr.h)
find_library(MPFR_LIBRARY mpfr)

# Find the GMP library
find_path(GMP_INCLUDE_DIR gmp.h)
find_library(GMP_LIBRARY gmp)

# Check if MPFR and GMP are present
if(NOT DEFINED ENV{SKIP_MPFR} AND MPFR_INCLUDE_DIR AND MPFR_LIBRARY AND GMP_INCLUDE_DIR AND GMP_LIBRARY)
    message(STATUS "Found MPFR and GMP libraries")

    # Add headers
    include_directories(
        ${MPFR_INCLUDE_DIR}
        ${GMP_INCLUDE_DIR}
    )

    # Link the libraries
    link_libraries(
        ${MPFR_LIBRARY}
        ${GMP_LIBRARY}
    )
else()
    message(WARNING "Could not find MPFR and/or GMP library. Skipping support for arbitrary precision.")

    # Set to skip
    add_compile_definitions(SKIP_MPFR=1)
endif()
