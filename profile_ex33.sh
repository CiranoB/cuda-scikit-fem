#!/usr/bin/env sh

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.venv/bin/python"}
EXAMPLE_PATH="$ROOT_DIR/docs/examples/ex33.py"
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/_findings/ex33_cprofile"}
START_REFINED=${START_REFINED:-5}
END_REFINED=${END_REFINED:-30}

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python nao encontrado em $PYTHON_BIN" >&2
    echo "Defina PYTHON_BIN=/caminho/para/python e execute novamente." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

CSV_PATH="$OUTPUT_DIR/results.csv"
printf '%s\n' 'refined_times,total_seconds,solver_direct_seconds,solve_spsolve_cpu_seconds,solver_percent_of_total' > "$CSV_PATH"

for refined in $(seq "$START_REFINED" "$END_REFINED"); do
    PROFILE_PATH="$OUTPUT_DIR/ex33_refined_${refined}.prof"
    LOG_PATH="$OUTPUT_DIR/ex33_refined_${refined}.log"

    echo "Profiling ex33.py com REFINED_TIMES=$refined"

    REFINED_TIMES="$refined" "$PYTHON_BIN" -m cProfile -o "$PROFILE_PATH" "$EXAMPLE_PATH" > "$LOG_PATH" 2>&1

    PROFILE_PATH="$PROFILE_PATH" REFINED="$refined" "$PYTHON_BIN" - <<'PY' >> "$CSV_PATH"
import os
import pstats
from pathlib import Path

profile_path = os.environ["PROFILE_PATH"]
refined = int(os.environ["REFINED"])
stats = pstats.Stats(profile_path)

total_seconds = stats.total_tt
solver_direct_seconds = 0.0
solve_spsolve_cpu_seconds = 0.0

for func, stat in stats.stats.items():
    filename, _line, name = func
    primitive_calls, total_calls, total_time, cumulative_time, callers = stat
    normalized = Path(filename).as_posix()

    if normalized.endswith("/skfem/sparse_solvers.py") and name == "solve_spsolve_cpu":
        solve_spsolve_cpu_seconds += cumulative_time

    if not normalized.endswith("/skfem/utils.py") or name != "solver":
        continue

    if any(Path(caller[0]).as_posix().endswith("/skfem/utils.py") and caller[2] == "solve_linear"
           for caller in callers):
        solver_direct_seconds += cumulative_time

solver_percent = 0.0 if total_seconds == 0 else (solver_direct_seconds / total_seconds) * 100.0

print(
    f"{refined},"
    f"{total_seconds:.6f},"
    f"{solver_direct_seconds:.6f},"
    f"{solve_spsolve_cpu_seconds:.6f},"
    f"{solver_percent:.2f}"
)
PY
done

echo "Resultados salvos em $CSV_PATH"
echo "Perfis brutos em $OUTPUT_DIR"