"""Sweep every GMRES-compatible (and NOT CG-compatible) documentation example
and compare, at growing mesh sizes, five GPU GMRES variants against the CPU
direct solver ``spsolve``.

What it does
------------
1. Reads ``_findings/solver_suitability.csv`` and selects the examples marked
   ``X`` in the GMRES column and *not* marked in the CG column, i.e. systems
   that are square/non-singular but not (verified) symmetric positive
   definite -- the structural reason CG is unsuitable and GMRES is used
   instead.
2. For each example ``docs/examples/exNN.py`` it runs a mesh-refinement sweep
   by increasing the ``INCREASE_REFINE_MESH`` environment variable (1 = the
   original mesh of the example).  Each ``(example, level)`` pair is executed
   in its own subprocess for isolation, so a crash or GPU out-of-memory in one
   configuration never aborts the whole campaign.
3. Every run is launched with ``GMRES_PRECOND_BENCHMARK=1`` which makes
   ``skfem.utils.solver_direct_scipy`` measure the *first* linear solve with
   the CPU ``spsolve`` reference and then time the end-to-end GPU GMRES path
   (host->device transfer, preconditioner build, solve, device->host readback)
   for the five preconditioner cases:

       none, jacobi, ilu0, block_jacobi, polynomial

   and print machine-parseable ``[gmresbench]`` / ``[gmrescase]`` lines.
4. The harness parses those lines and writes one combined CSV row per
   ``(example, refine level)``, plus a ``status.csv`` summarising which
   examples were swept, skipped, or stopped and why.

Stopping rule (per example)
---------------------------
The refinement level starts at ``START_LEVEL`` and is incremented by one until
any of the following happens:

* the subprocess reports a GPU/host out-of-memory condition;
* the subprocess exits with a non-zero status (e.g. runs out of memory while
  assembling, or ``spsolve`` fails);
* the subprocess exceeds ``PER_RUN_TIMEOUT_S`` wall-clock seconds (typically
  because the CPU ``spsolve`` baseline has become very slow -- which is exactly
  the regime we want to bound);
* ``ndofs`` reaches/exceeds ``DOF_CAP``;
* the CPU ``spsolve`` time alone exceeds ``SPSOLVE_CAP_S``;
* ``ndofs`` is identical to the previous level's (``no_mesh_growth``), which
  happens for examples whose mesh is not parameterized by
  ``INCREASE_REFINE_MESH`` (e.g. ex26, which inherits ex17's fixed mesh);
* every preconditioner case has already failed (``maxiter``/``timeout``) at a
  smaller mesh (``all_cases_disabled``) -- climbing further would only widen
  the CPU/GPU gap without producing any new convergence signal;
* the global ceiling ``MAX_LEVEL`` is reached.

Examples that do not perform a plain ``A x = b`` solve through the default
solver never emit a ``[gmresbench]`` line; they are detected at the first
level and skipped with reason ``no_linear_solve``.

Run
---
    PYTHONPATH=. uv run python _findings/gmres_all_examples/benchmark.py

Useful environment variables
-----------------------------
START_LEVEL          first INCREASE_REFINE_MESH value (default 1)
MAX_LEVEL            last INCREASE_REFINE_MESH value  (default 100)
PER_RUN_TIMEOUT_S    wall-clock cap per subprocess    (default 1800)
DOF_CAP              stop climbing once ndofs reaches this ceiling (default 5,000,000)
SPSOLVE_CAP_S        stop climbing once CPU spsolve alone exceeds this (default 300)
GMRES_CASE_TIMEOUT_S per-preconditioner solve budget  (default 60, passed on)
GMRES_MAXITER        GMRES iteration cap               (default 5000, passed on)
GMRES_RESTART        GMRES restart length (empty = cupyx default), passed on
GMRES_BLOCK_SIZE     block size for block-Jacobi      (default 8, passed on)
GMRES_POLY_DEGREE    Neumann polynomial degree        (default 3, passed on)
TOLERANCE            GMRES relative tolerance         (default 1e-5, passed on)
EXAMPLES              comma-separated override, e.g. "1,2,4" (default: all GMRES-only)
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITABILITY_CSV = REPO_ROOT / "_findings" / "solver_suitability.csv"
EXAMPLES_DIR = REPO_ROOT / "docs" / "examples"
OUTPUT_DIR = Path(os.getenv("GMRES_BENCHMARK_OUTPUT_DIR", Path(__file__).resolve().parent))
RESULTS_CSV = OUTPUT_DIR / "results.csv"
STATUS_CSV = OUTPUT_DIR / "status.csv"
RUN_LOGS_DIR = OUTPUT_DIR / "run_logs"

CASES = ("none", "jacobi", "ilu0", "block_jacobi", "polynomial")

START_LEVEL = int(os.getenv("START_LEVEL", "1"))
MAX_LEVEL = int(os.getenv("MAX_LEVEL", "100"))
PER_RUN_TIMEOUT_S = float(os.getenv("PER_RUN_TIMEOUT_S", "1800"))
DOF_CAP = int(os.getenv("DOF_CAP", "5000000"))
SPSOLVE_CAP_S = float(os.getenv("SPSOLVE_CAP_S", "300"))

# Environment forwarded to every example subprocess.  The benchmark knobs are
# read by skfem.utils at solve time; SKIP_VISUALISATION avoids GUI/file output.
FORWARDED_DEFAULTS = {
    "GMRES_CASE_TIMEOUT_S": os.getenv("GMRES_CASE_TIMEOUT_S", "60"),
    "GMRES_MAXITER": os.getenv("GMRES_MAXITER", "5000"),
    "GMRES_RESTART": os.getenv("GMRES_RESTART", ""),
    "GMRES_BLOCK_SIZE": os.getenv("GMRES_BLOCK_SIZE", "8"),
    "GMRES_POLY_DEGREE": os.getenv("GMRES_POLY_DEGREE", "3"),
    "TOLERANCE": os.getenv("TOLERANCE", "1e-5"),
}

_OOM_RE = re.compile(r"OutOfMemoryError|out of memory|CUDA_ERROR_OUT_OF_MEMORY",
                     re.IGNORECASE)


def gmres_only_examples() -> list[int]:
    """Return exercise numbers marked GMRES-compatible and not CG-compatible."""
    numbers: list[int] = []
    with SUITABILITY_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            is_gmres = (row.get("GMRES") or "").strip().upper() == "X"
            is_cg = (row.get("CG") or "").strip().upper() == "X"
            if is_gmres and not is_cg:
                numbers.append(int(row["exercise"]))
    return numbers


def parse_kv(line: str, tag: str) -> dict | None:
    """Parse a ``[tag] key=value key=value ...`` line into a dict."""
    prefix = f"[{tag}]"
    idx = line.find(prefix)
    if idx == -1:
        return None
    fields: dict = {}
    for token in line[idx + len(prefix):].split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        try:
            fields[key] = int(value)
        except ValueError:
            try:
                fields[key] = float(value)
            except ValueError:
                fields[key] = value
    return fields


def run_one(
    example: int,
    level: int,
    disabled_cases: set[str],
) -> tuple[dict | None, str, str]:
    """Run one (example, level) subprocess.

    Returns ``(parsed_row_or_None, stop_reason, raw_output)`` where
    ``stop_reason`` is empty when the sweep may continue.
    """
    example_file = EXAMPLES_DIR / f"ex{example:02d}.py"
    env = os.environ.copy()
    env.update(FORWARDED_DEFAULTS)
    env["INCREASE_REFINE_MESH"] = str(level)
    env["SKIP_VISUALISATION"] = "1"
    env["GMRES_PRECOND_BENCHMARK"] = "1"
    env["GMRES_BENCHMARK_DISABLED_CASES"] = ",".join(sorted(disabled_cases))
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"

    try:
        proc = subprocess.run(
            [sys.executable, str(example_file)],
            env=env,
            capture_output=True,
            text=True,
            timeout=PER_RUN_TIMEOUT_S,
            cwd=str(REPO_ROOT),
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\n" + (exc.stderr or "") \
            if isinstance(exc.stdout, str) else ""
        returncode = None
        timed_out = True

    header = None
    cases: dict[str, dict] = {}
    for line in output.splitlines():
        if "[gmresbench]" in line:
            header = parse_kv(line, "gmresbench")
        elif "[gmrescase]" in line:
            parsed = parse_kv(line, "gmrescase")
            if parsed and "case" in parsed:
                cases[str(parsed["case"])] = parsed

    # Decide whether the sweep for this example may continue.
    if timed_out:
        stop_reason = "timeout"
    elif returncode not in (0, None):
        stop_reason = "oom" if _OOM_RE.search(output) else f"exit_{returncode}"
    elif header is None:
        stop_reason = "no_linear_solve"
    elif _OOM_RE.search(output):
        stop_reason = "oom"
    else:
        stop_reason = ""

    if header is None:
        return None, stop_reason, output

    row: dict = {
        "example": example,
        "file": example_file.name,
        "refined": level,
        "ndofs": header.get("ndofs"),
        "nnz": header.get("nnz"),
        "spsolve_cpu_s": header.get("spsolve_cpu_s"),
        "transfer_s": header.get("transfer_s"),
        "tol": header.get("tol"),
        "maxiter": header.get("maxiter"),
        "restart": header.get("restart"),
    }
    spsolve_s = header.get("spsolve_cpu_s")
    for case in CASES:
        data = cases.get(case, {})
        total_s = data.get("total_s")
        row[f"{case}_status"] = data.get("status", "missing")
        row[f"{case}_precond_s"] = data.get("precond_s")
        row[f"{case}_solve_s"] = data.get("solve_s")
        row[f"{case}_readback_s"] = data.get("readback_s")
        row[f"{case}_total_s"] = total_s
        row[f"{case}_converged"] = data.get("converged")
        row[f"{case}_iters"] = data.get("iters")
        row[f"{case}_relres"] = data.get("relres")
        row[f"{case}_max_abs_diff"] = data.get("max_abs_diff")
        row[f"{case}_mean_abs_diff"] = data.get("mean_abs_diff")
        row[f"{case}_speedup_total"] = (
            spsolve_s / total_s
            if isinstance(spsolve_s, (int, float))
            and isinstance(total_s, (int, float)) and total_s > 0
            else None
        )
    return row, stop_reason, output


def result_fieldnames() -> list[str]:
    base = ["example", "file", "refined", "ndofs", "nnz", "spsolve_cpu_s",
            "transfer_s", "tol", "maxiter", "restart"]
    per_case = ["status", "precond_s", "solve_s", "readback_s", "total_s",
                "converged", "iters", "relres", "max_abs_diff",
                "mean_abs_diff", "speedup_total"]
    for case in CASES:
        base.extend(f"{case}_{field}" for field in per_case)
    return base


def main() -> None:
    override = os.getenv("EXAMPLES")
    if override:
        examples = [int(x) for x in override.split(",") if x.strip()]
    else:
        examples = gmres_only_examples()

    print(f"GMRES-only examples ({len(examples)}): {examples}", flush=True)
    print(f"levels {START_LEVEL}..{MAX_LEVEL}, per-run timeout "
          f"{PER_RUN_TIMEOUT_S:g}s, dof_cap={DOF_CAP:,}, "
          f"spsolve_cap={SPSOLVE_CAP_S:g}s, forwarded={FORWARDED_DEFAULTS}",
          flush=True)

    fieldnames = result_fieldnames()
    rows: list[dict] = []
    status: list[dict] = []
    campaign_start = time.perf_counter()

    for example in examples:
        example_file = EXAMPLES_DIR / f"ex{example:02d}.py"
        if not example_file.exists():
            status.append({"example": example, "levels_done": 0,
                           "max_ndofs": "", "reason": "missing_file"})
            print(f"ex{example:02d}: missing file, skipping", flush=True)
            continue

        levels_done = 0
        max_ndofs = 0
        prev_ndofs: int | None = None
        disabled_cases: set[str] = set()
        reason = f"reached_max_level_{MAX_LEVEL}"
        for level in range(START_LEVEL, MAX_LEVEL + 1):
            t0 = time.perf_counter()
            row, stop_reason, output = run_one(example, level, disabled_cases)
            dt = time.perf_counter() - t0
            run_log = RUN_LOGS_DIR / f"ex{example:02d}_L{level}.log"
            RUN_LOGS_DIR.mkdir(parents=True, exist_ok=True)
            run_log.write_text(output)

            if row is not None:
                rows.append(row)
                levels_done += 1
                ndofs = int(row.get("ndofs") or 0)
                max_ndofs = max(max_ndofs, ndofs)
                spsolve_s = row.get("spsolve_cpu_s")
                # Incrementally persist so partial campaigns are never lost.
                write_results(RESULTS_CSV, fieldnames, rows)
                summary = " ".join(
                    f"{c}={row.get(f'{c}_status')}/"
                    f"{row.get(f'{c}_total_s')}s" for c in CASES)
                print(f"ex{example:02d} L{level} ndofs={row.get('ndofs')} "
                      f"spsolve={row.get('spsolve_cpu_s')}s [{dt:.0f}s] {summary}",
                      flush=True)

                disabled_cases.update(
                    case for case in CASES
                    if row.get(f"{case}_status") in {"maxiter", "timeout"}
                )

                if not stop_reason and ndofs >= DOF_CAP:
                    stop_reason = "reached_dof_cap"
                elif not stop_reason and isinstance(spsolve_s, (int, float)) \
                        and spsolve_s > SPSOLVE_CAP_S:
                    stop_reason = "spsolve_cap_exceeded"
                elif not stop_reason and prev_ndofs is not None \
                        and ndofs == prev_ndofs:
                    stop_reason = "no_mesh_growth"
                elif not stop_reason and disabled_cases.issuperset(CASES):
                    # Every case already failed at a smaller mesh; climbing
                    # further only produces "skipped" rows with no signal.
                    stop_reason = "all_cases_disabled"
                prev_ndofs = ndofs
            else:
                print(f"ex{example:02d} L{level}: no result "
                    f"(reason={stop_reason or 'unknown'}) [{dt:.0f}s] "
                    f"log={run_log}",
                      flush=True)
                if output.strip():
                  print(f"--- ex{example:02d} L{level} subprocess output ---",
                      flush=True)
                  print(output.rstrip(), flush=True)
                  print(f"--- end ex{example:02d} L{level} subprocess output ---",
                      flush=True)

            if stop_reason:
                reason = stop_reason
                break

        status.append({"example": example, "levels_done": levels_done,
                       "max_ndofs": max_ndofs, "reason": reason})
        write_status(STATUS_CSV, status)

    total = time.perf_counter() - campaign_start
    print(f"\nDone in {total:.0f}s. {len(rows)} rows -> {RESULTS_CSV}",
          flush=True)
    print(f"Status -> {STATUS_CSV}", flush=True)


def write_results(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_status(path: Path, status: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["example", "levels_done", "max_ndofs", "reason"])
        writer.writeheader()
        writer.writerows(status)


if __name__ == "__main__":
    main()
