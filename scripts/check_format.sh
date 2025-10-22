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
CLANG_FORMAT=19.1.3

# Fix yapf version
YAPF=0.43.0

FAILED="\033[91m[FAILED]\033[0m"
OK="\033[92m[  OK  ]\033[0m"
WARNING="\033[93m[WARNIN]\033[0m"

# Install
if [[ -n ${INSTALL} ]]; then
  pip install clang-format==${CLANG_FORMAT} \
              yapf==${YAPF} \
              ruff
fi

# Check clang-format version
if [[ $(clang-format --version 2>/dev/null | awk "{ print (\$3 == \"${CLANG_FORMAT}\") }") != 1 ]]; then
  echo "clang-format==${CLANG_FORMAT} is required" >&2
  exit 1
fi

# Check yapf version
if [[ $(yapf --version 2>/dev/null | awk "{ print (\$2 == \"${YAPF}\") }") != 1 ]]; then
  echo "yapf==${YAPF} is required" >&2
  exit 1
fi

# Check ruff
if ! ruff --version >/dev/null; then
  echo "ruff is required" >&2
  exit 1
fi

# Check cpp files
CPP_CMD='clang-format --style=google'
CPP_FILES=$(git ls-files --exclude-per-directory=.gitignore | \
            grep -iE $'(\.cpp|\.hpp)' | \
            parallel -P8 'echo {} $(file {}) |
                          grep -E "(C|C\+\+) source" | awk "{print \$1}"')
CPP_FAILED=$(echo -n ${CPP_FILES} | tr ' ' '\n' | parallel -P8 "
  if [[ \$(${CPP_CMD} --output-replacements-xml {} | wc -l) -gt 3 ]];
   then
     echo -ne \"${FAILED} \" >&2;
     echo {};
   else
     echo -ne \"${OK} \" >&2;
   fi;
   echo \"(format)\" {} >&2" | tr '\n' ' ')

# Check python files
PYTHON_CMD='yapf --style=google'
PYTHON_FILES=$(git ls-files --exclude-per-directory=.gitignore | \
               grep -v 'README.md' | \
               parallel -P8 'echo {} $(file {}) |
                             grep -E "Python script" | awk "{print \$1}"')
PYTHON_FAILED=$(echo -n ${PYTHON_FILES} | tr ' ' '\n' | parallel -P8 "
  if [[ \$(${PYTHON_CMD} -d {} | wc -l) -gt 0 ]];
   then
     echo -ne \"${FAILED} \" >&2;
     echo {};
   else
     echo -ne \"${OK} \" >&2;
   fi;
   echo \"(format)\" {} >&2" | tr '\n' ' ')

if [[ -n "${CPP_FAILED}" ]]; then
  echo -e "${FAILED} Some C/C++ files are not properly formatted." \
          "Run:\n\n           ${CPP_CMD} -i ${CPP_FAILED}\n" >&2
fi

if [[ -n "${PYTHON_FAILED}" ]]; then
  echo -e "${FAILED} Some Python files are not properly formatted." \
          "Run:\n\n           ${PYTHON_CMD} -i ${PYTHON_FAILED}\n" >&2
fi

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
ruff check tnco/ tests/ >&2
LINTING_FAILED=$?

# Try to fix the errors
if [[ -n ${FIX} ]]; then
  if [[ -n "${CPP_FAILED}" ]]; then
    echo -en "\033[92m[FIXING]\033[0m C++: "${CPP_FAILED}"\n"
    ${CPP_CMD} -i ${CPP_FAILED}
  fi
  if [[ -n "${PYTHON_FAILED}" ]]; then
    echo -en "\033[92m[FIXING]\033[0m Python: "${PYTHON_FAILED}"\n"
    ${PYTHON_CMD} -i ${PYTHON_FAILED}
  fi
  if [[ "${LINTING_FAILED}" > 0 ]]; then
    echo -en "\033[92m[FIXING]\033[0m Ruff\n"
    ruff check tnco/ tests/ --fix
  fi
  exit 0
fi

# Raise error
if [[ -n "${CPP_FAILED}" || \
      -n "${PYTHON_FAILED}" || \
      -n "${TRAIL_FAILED}" || \
      "${LINTING_FAILED}" > 0 ]]; then
  exit 1
fi
