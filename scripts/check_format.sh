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
    echo "Usage: $0 [--fix] [--install] [-h|--help]"
    exit 1
}

# Parse options using getopt
OPTIONS=$(getopt -o h --long help,install,fix -- "$@")
if [[ $? -ne 0 ]]; then
    usage
fi

while [[ $# -ne 0 ]]; do
    case "$1" in
        --install)
            INSTALL=true
            shift
            ;;
        --fix)
            FIX=true
            shift
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

# Fix clang-format version
CLANG_FORMAT_VERSION=19.1.3

# Fix yapf version
YAPF_VERSION=0.43.0

# Fix isort version
ISORT_VERSION=7.0.0

# Fix ruff version
RUFF_VERSION=0.14.1

FAILED="\033[91m[FAILED]\033[0m"
OK="\033[92m[  OK  ]\033[0m"
WARNING="\033[93m[WARNIN]\033[0m"

# Install
if [[ -n ${INSTALL} ]]; then
  pip install clang-format==${CLANG_FORMAT_VERSION} \
              yapf==${YAPF_VERSION} \
              isort==${ISORT_VERSION} \
              ruff==${RUFF_VERSION}
fi

# Check clang-format version
if [[ $(clang-format --version 2>/dev/null | awk "{ print (\$3 == \"${CLANG_FORMAT_VERSION}\") }") != 1 ]]; then
  echo "clang-format==${CLANG_FORMAT_VERSION} is required" >&2
  exit 1
fi

# Check yapf version
if [[ $(yapf --version 2>/dev/null | awk "{ print (\$2 == \"${YAPF_VERSION}\") }") != 1 ]]; then
  echo "yapf==${YAPF_VERSION} is required" >&2
  exit 1
fi

# Check isort version
if [[ $(isort --version 2>/dev/null | grep -i version | awk "{ print (\$2 == \"${ISORT_VERSION}\") }") != 1 ]]; then
  echo "isort==${ISORT_VERSION} is required" >&2
  exit 1
fi

# Check ruff version
if [[ $(ruff --version 2>/dev/null | awk "{ print (\$2 == \"${RUFF_VERSION}\") }") != 1 ]]; then
  echo "isort==${RUFF_VERSION} is required" >&2
  exit 1
fi

# Check ruff
if ! ruff --version >/dev/null; then
  echo "ruff is required" >&2
  exit 1
fi

# Check cpp files
CLANG_FORMAT_CMD='clang-format --style=google'
CPP_FILES=$(git ls-files --exclude-per-directory=.gitignore | \
            grep -iE $'(\.cpp|\.hpp)' | \
            parallel 'echo {} $(file {}) |
                      grep -E "(C|C\+\+) source" | awk "{print \$1}"')
CLANG_FORMAT_FAILED=$(echo -n ${CPP_FILES} | tr ' ' '\n' | parallel "
  if [[ \$(${CLANG_FORMAT_CMD} --output-replacements-xml {} | wc -l) -gt 3 ]];
   then
     echo -ne \"${FAILED} \" >&2;
     echo {};
   else
     echo -ne \"${OK} \" >&2;
   fi;
   echo \"(format)\" {} >&2" | tr '\n' ' ')

# Check python files
YAPF_CMD='yapf --style=google'
PYTHON_FILES=$(git ls-files --exclude-per-directory=.gitignore | \
               grep -v 'README.md' | \
               parallel 'echo {} $(file {}) |
                         grep -E "Python script" | awk "{print \$1}"')
YAPF_FAILED=$(echo -n ${PYTHON_FILES} | tr ' ' '\n' | parallel "
  if [[ \$(${YAPF_CMD} -d {} | wc -l) -gt 0 ]];
   then
     echo -ne \"${FAILED} \" >&2;
     echo {};
   else
     echo -ne \"${OK} \" >&2;
   fi;
   echo \"(format)\" {} >&2" | tr '\n' ' ')

# Check imports
ISORT_CMD='isort'
ISORT_FAILED=$(echo -n ${PYTHON_FILES} | tr ' ' '\n' | parallel "
  if [[ \$(${ISORT_CMD} -c {} 2>&1 | wc -l) -gt 0 ]];
   then
     echo -ne \"${FAILED} \" >&2;
     echo {};
   else
     echo -ne \"${OK} \" >&2;
   fi;
   echo \"(isort)\" {} >&2" | tr '\n' ' ')

# Check for files with rows too long
LONG_ROWS=$(git ls-files | grep -E $"\.(cpp|hpp|py)" | \
               parallel 'echo {} $(($(\
                if [[ -s {} ]]; then \
                  cat {} | awk "{ print length }" | sort -g | \
                                                    tail -n 1; \
                else \
                  echo 0; \
                fi) > 80))' | awk '$NF != 0 { $NF=0; print $1 }')

for FILE in ${LONG_ROWS}; do
  echo -e ${WARNING} '(long-rows)' $FILE >&2
done

# Check for trailing whitespaces
TRAIL_FAILED=$(git ls-files | parallel 'echo {} $(cat {} | \
                              grep '[[:blank:]]'$ | wc -l)' | \
                              awk '$NF > 0 { $NF=""; print $1 }')


RED_BLOCK='\033[41m$\033[0m'
if [[ -n ${TRAIL_FAILED} ]]; then
  parallel "echo @@@@ {}; cat {} | grep --color=always -n '[[:blank:]]'$" ::: ${TRAIL_FAILED} | \
                              awk "{
                                if(\$1 == \"@@@@\")
                                  print \"${FAILED} \" \$2
                                else
                                  print \$0\"${RED_BLOCK}\"
                              }"
fi

# Linting
ruff check tnco/ tests/ >/dev/null
LINTING_FAILED=$?
if [[ ${LINTING_FAILED} -eq 0 ]]; then
  echo -e "${OK} (ruff) All tests passed."
else
  echo -e "${FAILED} (ruff) Linting failed."
fi

if [[ -n "${CLANG_FORMAT_FAILED}" ]]; then
  echo -e "${FAILED} Some C/C++ files are not properly formatted." \
          "Run:\n\n           ${CLANG_FORMAT_CMD} -i ${CLANG_FORMAT_FAILED}\n" >&2
fi

if [[ -n "${YAPF_FAILED}" ]]; then
  echo -e "${FAILED} Some Python files are not properly formatted." \
          "Run:\n\n           ${YAPF_CMD} -i ${YAPF_FAILED}\n" >&2
fi

if [[ -n "${ISORT_FAILED}" ]]; then
  echo -e "${FAILED} Imports in some Python files are out of order." \
          "Run:\n\n           ${ISORT_CMD} ${ISORT_FAILED}\n" >&2
fi

# Try to fix the errors
if [[ -n ${FIX} ]]; then
  if [[ -n "${CLANG_FORMAT_FAILED}" ]]; then
    echo -en "\033[92m[FIXING]\033[0m C++: "${CLANG_FORMAT_FAILED}"\n"
    ${CLANG_FORMAT_CMD} -i ${CLANG_FORMAT_FAILED}
  fi
  if [[ -n "${YAPF_FAILED}" ]]; then
    echo -en "\033[92m[FIXING]\033[0m Python: "${YAPF_FAILED}"\n"
    ${YAPF_CMD} -i ${YAPF_FAILED}
  fi
  if [[ -n "${ISORT_FAILED}" ]]; then
    echo -en "\033[92m[FIXING]\033[0m Python: "${ISORT_FAILED}"\n"
    ${ISORT_CMD} ${ISORT_FAILED} >/dev/null
  fi
  if [[ "${LINTING_FAILED}" > 0 ]]; then
    echo -en "\033[92m[FIXING]\033[0m Ruff\n"
    ruff check tnco/ tests/ --fix
  fi
  exit 0
fi

# Raise error
if [[ -n "${CLANG_FORMAT_FAILED}" || \
      -n "${YAPF_FAILED}" || \
      -n "${ISORT_FAILED}" || \
      -n "${TRAIL_FAILED}" || \
      "${LINTING_FAILED}" > 0 ]]; then
  exit 1
fi
