#!/usr/bin/env bash
# Run the documented CG benchmark sweep without replacing the reference data.
#
# Usage from any directory:
#   docs/examples/run_cg_compatible_examples.sh
#
# Optional environment variables are forwarded to the harness, e.g.:
#   EXAMPLES=3,6,7 MAX_LEVEL=10 docs/examples/run_cg_compatible_examples.sh
#   OUTPUT_ROOT=/scratch/cg-runs docs/examples/run_cg_compatible_examples.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-"${SCRIPT_DIR}/cg_all_examples_results"}
RESULTS_DIR="${OUTPUT_ROOT}/$(date +%Y%m%dT%H%M%S)"

mkdir -p "${RESULTS_DIR}"

# Each refinement is isolated by benchmark.py and may run for at most 1000 s.
export PER_RUN_TIMEOUT_S=${PER_RUN_TIMEOUT_S:-1000}
export CG_BENCHMARK_OUTPUT_DIR="${RESULTS_DIR}"

{
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'per_run_timeout_s=%s\n' "${PER_RUN_TIMEOUT_S}"
    for variable in START_LEVEL MAX_LEVEL DOF_CAP SPSOLVE_CAP_S EXAMPLES \
        CG_CASE_TIMEOUT_S CG_MAXITER CG_BLOCK_SIZE CG_POLY_DEGREE TOLERANCE; do
        printf '%s=%s\n' "${variable}" "${!variable:-<default>}"
    done
} > "${RESULTS_DIR}/run.env"

cd "${REPO_ROOT}"
PYTHONPATH=. uv run python _findings/cg_all_examples/benchmark.py \
    2>&1 | tee "${RESULTS_DIR}/campaign.log"

if [[ -f "${RESULTS_DIR}/results.csv" ]]; then
    printf '\nResults: %s\n' "${RESULTS_DIR}/results.csv"
else
    printf '\nResults: no measurements were produced\n'
fi
printf 'Status:  %s\n' "${RESULTS_DIR}/status.csv"
printf 'Log:     %s\n' "${RESULTS_DIR}/campaign.log"