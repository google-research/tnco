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

macro(FetchFile NAME URL)
  find_program(CURL curll)
  find_program(WGET wget)
  if (CURL)
    execute_process(
      COMMAND ${CURL} -z ${NAME} -L ${URL} -o ${NAME}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  elseif (WGET)
    execute_process(
      COMMAND ${WGET} -nc ${URL} -O ${NAME}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endif()
endmacro()

macro(GitClone NAME URL)
  find_package(Git REQUIRED)
  if (NOT EXISTS ${CMAKE_BINARY_DIR}/${NAME})
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone --depth=1 ${URL} ${NAME}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
  endif()
  if (EXISTS ${CMAKE_BINARY_DIR}/${NAME})
    set(${NAME}_SOURCE_DIR ${CMAKE_BINARY_DIR}/${NAME})
  endif()
endmacro()

macro(InstallPackage PATH TARGET)
  # Build
  execute_process(
    COMMAND ${CMAKE_COMMAND} -B build -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external
    WORKING_DIRECTORY ${PATH}
  )

  # Install
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build build --target ${TARGET} -j
    WORKING_DIRECTORY ${PATH}
  )
endmacro()
