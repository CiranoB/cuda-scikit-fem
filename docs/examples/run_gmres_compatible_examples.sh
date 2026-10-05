#!/usr/bin/env bash
# Run the documented GMRES benchmark sweep without replacing the reference data.
#
# Usage from any directory:
#   docs/examples/run_gmres_compatible_examples.sh
#
# Optional environment variables are forwarded to the harness, e.g.:
#   EXAMPLES=1,2,4 MAX_LEVEL=10 docs/examples/run_gmres_compatible_examples.sh
#   OUTPUT_ROOT=/scratch/gmres-runs docs/examples/run_gmres_compatible_examples.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-"${SCRIPT_DIR}/gmres_all_examples_results"}
RESULTS_DIR="${OUTPUT_ROOT}/$(date +%Y%m%dT%H%M%S)"

mkdir -p "${RESULTS_DIR}"

# Each refinement is isolated by benchmark.py and may run for at most 1000 s.
export PER_RUN_TIMEOUT_S=${PER_RUN_TIMEOUT_S:-1000}
export HALF_PRECISION=1
export GMRES_BENCHMARK_OUTPUT_DIR="${RESULTS_DIR}"

{
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'per_run_timeout_s=%s\n' "${PER_RUN_TIMEOUT_S}"
    for variable in HALF_PRECISION START_LEVEL MAX_LEVEL DOF_CAP SPSOLVE_CAP_S EXAMPLES \
        GMRES_CASE_TIMEOUT_S GMRES_MAXITER GMRES_RESTART GMRES_BLOCK_SIZE \
        GMRES_POLY_DEGREE TOLERANCE; do
        printf '%s=%s\n' "${variable}" "${!variable:-<default>}"
    done
} > "${RESULTS_DIR}/run.env"

cd "${REPO_ROOT}"
PYTHONPATH=. uv run python _findings/gmres_all_examples/benchmark.py \
    2>&1 | tee "${RESULTS_DIR}/campaign.log"

if [[ -f "${RESULTS_DIR}/results.csv" ]]; then
    printf '\nResults: %s\n' "${RESULTS_DIR}/results.csv"
else
    printf '\nResults: no measurements were produced\n'
fi
printf 'Status:  %s\n' "${RESULTS_DIR}/status.csv"
printf 'Log:     %s\n' "${RESULTS_DIR}/campaign.log"
