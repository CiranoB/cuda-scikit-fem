#!/usr/bin/env bash
# Run the documented GMRES benchmark sweep without replacing the reference data.
#
# Usage from any directory:
#   docs/examples/run_gmres_compatible_examples.sh
#
# Optional environment variables are forwarded to the harness, e.g.:
#   EXAMPLES=1,2,4 MAX_LEVEL=10 docs/examples/run_gmres_compatible_examples.sh
#   OUTPUT_ROOT=/scratch/gmres-runs docs/examples/run_gmres_compatible_examples.sh
#   REPEATS=10 (default) full campaigns are run; mean/std are written to aggregate.csv

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-"${SCRIPT_DIR}/gmres_all_examples_results"}
RESULTS_DIR="${OUTPUT_ROOT}/$(date +%Y%m%dT%H%M%S)"
REPEATS=${REPEATS:-10}

if [[ ! "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'REPEATS must be a positive integer\n' >&2
    exit 2
fi

mkdir -p "${RESULTS_DIR}"

# Each refinement is isolated by benchmark.py and may run for at most 3600 s.
export PER_RUN_TIMEOUT_S=${PER_RUN_TIMEOUT_S:-3600}
export HALF_PRECISION=${HALF_PRECISION:-0}
# Per-preconditioner GPU solve budget; PER_RUN_TIMEOUT_S must exceed 5 cases x this + CPU cases + spsolve.
export GMRES_CASE_TIMEOUT_S=${GMRES_CASE_TIMEOUT_S:-500}
# Large enough that the per-case time budget, not the iteration cap, ends a run.
export GMRES_MAXITER=${GMRES_MAXITER:-100000}
# Restart 20 stagnates on ex04/ex18/ex34; they only converged at 100.
export GMRES_RESTART=${GMRES_RESTART:-100}

{
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'per_run_timeout_s=%s\n' "${PER_RUN_TIMEOUT_S}"
    printf 'repeats=%s\n' "${REPEATS}"
    for variable in HALF_PRECISION START_LEVEL MAX_LEVEL DOF_CAP SPSOLVE_CAP_S EXAMPLES \
        GMRES_CASE_TIMEOUT_S GMRES_CPU_CASE_TIMEOUT_S GMRES_BENCH_CPU GMRES_MAXITER \
        GMRES_RESTART GMRES_BLOCK_SIZE GMRES_POLY_DEGREE TOLERANCE; do
        printf '%s=%s\n' "${variable}" "${!variable:-<default>}"
    done
} > "${RESULTS_DIR}/run.env"

cd "${REPO_ROOT}"
for run in $(seq 1 "${REPEATS}"); do
    RUN_DIR="${RESULTS_DIR}/run_${run}"
    mkdir -p "${RUN_DIR}"
    printf '\n=== Repetition %s/%s ===\n' "${run}" "${REPEATS}"
    GMRES_BENCHMARK_OUTPUT_DIR="${RUN_DIR}" PYTHONPATH=. \
        uv run python _findings/gmres_all_examples/benchmark.py \
        2>&1 | tee "${RUN_DIR}/campaign.log"
done

RESULT_FILES=("${RESULTS_DIR}"/run_*/results.csv)
if [[ -f "${RESULT_FILES[0]}" ]]; then
    PYTHONPATH=. uv run python _findings/cg_all_examples/aggregate_repeats.py "${RESULTS_DIR}" \
        2>&1 | tee "${RESULTS_DIR}/aggregate.log"

    printf '\nResults: %s/run_*/results.csv\n' "${RESULTS_DIR}"
    printf 'Mean/std: %s\n' "${RESULTS_DIR}/aggregate.csv"
else
    printf '\nResults: no measurements were produced\n'
fi
printf 'Status:  %s/run_*/status.csv\n' "${RESULTS_DIR}"
printf 'Logs:    %s/run_*/campaign.log\n' "${RESULTS_DIR}"
