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

include(FetchContent)

find_package(Boost)

if (NOT Boost_FOUND)
  if (DEFINED ENV{DOWNLOAD_DEPS})
    message(STATUS "Downloads 'boost'")

    # Download boost
    FetchContent_Declare(
      boost
      URL https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(boost)

    # Check download
    if (NOT boost_SOURCE_DIR)
      message(FATAL_ERROR "'boost' not found.")
    endif()

    # Add boost to include
    include_directories(${boost_SOURCE_DIR})

  else()
    message(FATAL_ERROR "'boost' not found.")
  endif()

else()
  include_directories(${Boost_INCLUDE_DIRS})
endif()
