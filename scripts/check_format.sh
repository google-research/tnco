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
OPTIONS=$(getopt -o h --long help,config,fix -- "$@")
if [[ $? -ne 0 ]]; then
    usage
fi

while [[ $# -ne 0 ]]; do
    case "$1" in
        --config)
            CONFIG=true
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

# Fix versions
get_version() {
  grep "$1==" pyproject.toml | sed -E "s/.*$1==([^ \"]+).*/\1/"
}
CLANG_FORMAT_VERSION=$(get_version clang-format)
YAPF_VERSION=$(get_version yapf)
ISORT_VERSION=$(get_version isort)
RUFF_VERSION=$(get_version ruff)
DOCFORMATTER_VERSION=$(get_version docformatter)

FAILED="\033[91m[FAILED]\033[0m"
OK="\033[92m[  OK  ]\033[0m"
WARNING="\033[93m[WARNIN]\033[0m"

# Install
if [[ -n ${CONFIG} ]]; then
  echo clang-format==${CLANG_FORMAT_VERSION} \
               yapf==${YAPF_VERSION} \
              isort==${ISORT_VERSION} \
               ruff==${RUFF_VERSION} \
        docformatter==${DOCFORMATTER_VERSION}
  exit 0
fi

# Check clang-format version
CF_VER_CHK=$(clang-format --version 2>/dev/null | \
  awk "{ print (\$3 == \"${CLANG_FORMAT_VERSION}\") }")
if [[ ${CF_VER_CHK} != 1 ]]; then
  echo "clang-format==${CLANG_FORMAT_VERSION} is required" >&2
  exit 1
fi

# Check yapf version
YAPF_VER_CHK=$(yapf --version 2>/dev/null | \
  awk "{ print (\$2 == \"${YAPF_VERSION}\") }")
if [[ ${YAPF_VER_CHK} != 1 ]]; then
  echo "yapf==${YAPF_VERSION} is required" >&2
  exit 1
fi

# Check isort version
ISORT_VER_CHK=$(isort --version 2>/dev/null | grep -i version | \
  awk "{ print (\$2 == \"${ISORT_VERSION}\") }")
if [[ ${ISORT_VER_CHK} != 1 ]]; then
  echo "isort==${ISORT_VERSION} is required" >&2
  exit 1
fi

# Check ruff version
RUFF_VER_CHK=$(ruff --version 2>/dev/null | \
  awk "{ print (\$2 == \"${RUFF_VERSION}\") }")
if [[ ${RUFF_VER_CHK} != 1 ]]; then
  echo "ruff==${RUFF_VERSION} is required" >&2
  exit 1
fi

# Check docformatter version
DOCFORMATTER_VER_CHK=$(docformatter --version 2>/dev/null | \
  awk "{ print (\$2 == \"${DOCFORMATTER_VERSION}\") }")
if [[ ${DOCFORMATTER_VER_CHK} != 1 ]]; then
  echo "docformatter==${DOCFORMATTER_VERSION} is required" >&2
  exit 1
fi

# Check ruff
if ! ruff --version >/dev/null 2>&1; then
  echo "ruff is required" >&2
  exit 1
fi

# Check cpp files
CLANG_FORMAT_CMD='clang-format --style=google'
CPP_FILES=$(git ls-files --exclude-per-directory=.gitignore | \
            grep -iE '\.(cpp|hpp)$' | \
            parallel 'echo {} $(file {}) | \
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
               parallel 'echo {} $(file {}) | \
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

# Check docstrings
DOCFORMATTER_CMD='docformatter'
DOCFORMATTER_FAILED=$(echo -n ${PYTHON_FILES} | tr ' ' '\n' | grep -v '^tests/' | parallel "
  if [[ \$(${DOCFORMATTER_CMD} -c {} 2>&1 | wc -l) -gt 0 ]];
   then
     echo -ne \"${FAILED} \" >&2;
     echo {};
   else
     echo -ne \"${OK} \" >&2;
   fi;
   echo \"(docformatter)\" {} >&2" | tr '\n' ' ')

# Check for files with rows too long
LONG_ROWS=$(git ls-files | grep -E '\.(cpp|hpp|py)$' | \
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
                              grep "[[:blank:]]$" | wc -l)' | \
                              awk '$NF > 0 { $NF=""; print $1 }')

RED_BLOCK='\033[41m$\033[0m'
if [[ -n ${TRAIL_FAILED} ]]; then
  parallel "echo @@@@ {}; cat {} | grep --color=always -n '[[:blank:]]$'" \
    ::: ${TRAIL_FAILED} | \
    awk "{
      if(\$1 == \"@@@@\")
        print \"${FAILED} \" \$2
      else
        print \$0\"${RED_BLOCK}\"
    }"
fi

# Linting
ruff check tnco/ tests/ | grep -vi 'all checks passed' >&2
LINTING_FAILED=${PIPESTATUS[0]}
if [[ ${LINTING_FAILED} -eq 0 ]]; then
  echo -e "${OK} (ruff) All checks passed."
else
  echo -e "${FAILED} (ruff) Linting failed."
fi

if [[ -n "${CLANG_FORMAT_FAILED}" ]]; then
  echo -e "${FAILED} Some C/C++ files are not properly formatted." \
          "Run:\n\n          ${CLANG_FORMAT_CMD} -i ${CLANG_FORMAT_FAILED}\n" \
          >&2
fi

if [[ -n "${YAPF_FAILED}" ]]; then
  echo -e "${FAILED} Some Python files are not properly formatted." \
          "Run:\n\n          ${YAPF_CMD} -i ${YAPF_FAILED}\n" >&2
fi

if [[ -n "${ISORT_FAILED}" ]]; then
  echo -e "${FAILED} Imports in some Python files are out of order." \
          "Run:\n\n          ${ISORT_CMD} --overwrite-in-place ${ISORT_FAILED}\n" >&2
fi

if [[ -n "${DOCFORMATTER_FAILED}" ]]; then
  echo -e "${FAILED} Docstrings in some Python files are not properly formatted." \
          "Run:\n\n          ${DOCFORMATTER_CMD} -i ${DOCFORMATTER_FAILED}\n" >&2
fi

# Try to fix the errors
if [[ -n ${FIX} ]]; then
  if [[ -n "${CLANG_FORMAT_FAILED}" ]]; then
    echo -en "\033[92m[FIXING]\033[0m C++: ${CLANG_FORMAT_FAILED}\n"
    ${CLANG_FORMAT_CMD} -i ${CLANG_FORMAT_FAILED}
  fi
  if [[ -n "${YAPF_FAILED}" ]]; then
    echo -en "\033[92m[FIXING]\033[0m Python: ${YAPF_FAILED}\n"
    ${YAPF_CMD} -i ${YAPF_FAILED}
  fi
  if [[ -n "${ISORT_FAILED}" ]]; then
    echo -en "\033[92m[FIXING]\033[0m isort: ${ISORT_FAILED}\n"
    ${ISORT_CMD} ${ISORT_FAILED}
  fi
  if [[ -n "${DOCFORMATTER_FAILED}" ]]; then
    echo -en "\033[92m[FIXING]\033[0m docformatter: ${DOCFORMATTER_FAILED}\n"
    ${DOCFORMATTER_CMD} -i ${DOCFORMATTER_FAILED}
  fi
  if [[ "${LINTING_FAILED}" -gt 0 ]]; then
    echo -en "\033[92m[FIXING]\033[0m Ruff\n"
    ruff check tnco/ tests/ --fix
  fi
  exit 0
fi

# Raise error
if [[ -n "${CLANG_FORMAT_FAILED}" || \
      -n "${YAPF_FAILED}" || \
      -n "${ISORT_FAILED}" || \
      -n "${DOCFORMATTER_FAILED}" || \
      -n "${TRAIL_FAILED}" || \
      "${LINTING_FAILED}" -gt 0 ]]; then
  exit 1
fi
