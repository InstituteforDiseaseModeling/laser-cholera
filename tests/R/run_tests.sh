#!/usr/bin/env bash
# Run the R-vs-Python parity tests for calc_model_likelihood.
#
# Prerequisites:
#   - R (Rscript on PATH)
#   - R packages: testthat, reticulate, MOSAIC
#       Rscript -e 'install.packages(c("testthat", "reticulate"))'
#       Rscript -e 'remotes::install_github("InstituteforDiseaseModeling/MOSAIC-pkg")'
#   - The project venv at .venv/ with laser-cholera installed
#       (uv pip install -e . from the repo root)
#
# Invocation:
#   - From the repo root:    tests/R/run_tests.sh
#   - From the tests/R dir:  ./run_tests.sh

set -euo pipefail

# Resolve repo root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# Force reticulate onto the project's editable venv. RETICULATE_PYTHON takes
# precedence over `reticulate::use_python()`, so we have to override it here
# rather than from inside R if the user has it pointing elsewhere (e.g. at an
# `r-mosaic` venv with a stale, non-editable laser-cholera snapshot).
if [ -x "${REPO_ROOT}/.venv/bin/python3" ]; then
  export RETICULATE_PYTHON="${REPO_ROOT}/.venv/bin/python3"
fi

echo "================================================================"
echo "R-vs-Python parity tests for calc_model_likelihood"
echo "================================================================"
echo ""
echo "Repo root          : ${REPO_ROOT}"
echo "Test directory     : ${SCRIPT_DIR}"
echo ""
echo "Versions:"
Rscript -e 'cat("  R                : ", R.version.string, "\n", sep = "")'
Rscript -e 'cat("  MOSAIC           : ", as.character(packageVersion("MOSAIC")), "\n", sep = "")' \
  || { echo "  MOSAIC           : NOT INSTALLED — see prerequisites at the top of this script." >&2; exit 2; }
Rscript -e 'cat("  reticulate       : ", as.character(packageVersion("reticulate")), "\n", sep = "")' \
  || { echo "  reticulate       : NOT INSTALLED — see prerequisites at the top of this script." >&2; exit 2; }
Rscript -e 'cat("  testthat         : ", as.character(packageVersion("testthat")), "\n", sep = "")' \
  || { echo "  testthat         : NOT INSTALLED — see prerequisites at the top of this script." >&2; exit 2; }

if [ -x "${REPO_ROOT}/.venv/bin/python3" ]; then
  LC_VERSION="$("${REPO_ROOT}/.venv/bin/python3" -c \
    'import laser.cholera; print(getattr(laser.cholera, "__version__", "unknown"))' 2>/dev/null \
    || echo "NOT INSTALLED")"
  echo "  laser-cholera    : ${LC_VERSION}  (from ${REPO_ROOT}/.venv/bin/python3)"
else
  LC_VERSION="$(python3 -c \
    'import laser.cholera; print(getattr(laser.cholera, "__version__", "unknown"))' 2>/dev/null \
    || echo "NOT INSTALLED")"
  echo "  laser-cholera    : ${LC_VERSION}  (from system python3 — no .venv detected)"
fi

if [ "${LC_VERSION}" = "NOT INSTALLED" ]; then
  echo "" >&2
  echo "laser-cholera is not importable from the Python interpreter that reticulate will use." >&2
  echo "Install with: uv pip install -e ." >&2
  exit 2
fi

echo ""
echo "================================================================"
echo "Running testthat suite (tests/R/)"
echo "================================================================"

# Invoke testthat with a fail-loud reporter; exit code matches the testthat
# result so this script can gate CI.
Rscript -e '
suppressPackageStartupMessages(library(testthat))
res <- testthat::test_dir("tests/R", reporter = "summary", stop_on_failure = TRUE)
'
