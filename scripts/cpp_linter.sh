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

# Define the usage function
usage() {
    echo "Usage: $0 [--install] [-h|--help]"
    exit 1
}

# Parse options using getopt
OPTIONS=$(getopt -o h --long help,build:,install -- "$@")
if [[ $? -ne 0 ]]; then
    usage
fi

while [[ $# -ne 0 ]]; do
    case "$1" in
        --install)
            INSTALL=true
            shift
            ;;
        --build)
            BUILD=$2
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            usage
            ;;
    esac
done

# Fix clang-tidy version
CLANG_TIDY=21.1.1

# Install
if [[ -n ${INSTALL} ]]; then
  pip install clang-tidy==${CLANG_TIDY}
fi

# Check version
if [[ $(clang-tidy --version | grep -i 'llvm version' | awk '{ print $NF }' 2>/dev/null) != ${CLANG_TIDY} ]]; then
  echo "clang-tidy==${CLANG_TIDY} is required" >&2
fi

# Get location of build
BUILD=${BUILD:-build/}
export BUILD

check() {
  FAILED="\033[91m[FAILED]\033[0m"
  OK="\033[92m[  OK  ]\033[0m"

  if clang-tidy -p ${BUILD} "$1" >/dev/null 2>/dev/null; then
    echo -e "\r${OK} $1" 2>/dev/null
    return 0
  else
    echo -e "\r${FAILED} $1" 2>/dev/null
    return 1
  fi
}
export -f check

# Call linter
git ls-files -z '*.cpp' '*.hpp' | parallel -0 --halt now,fail,0 --lb check {}
