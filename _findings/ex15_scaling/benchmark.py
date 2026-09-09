"""Logarithmic CPU-versus-GPU CG sweep for the one-dimensional Example 15.

Run with:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "docs" / "examples" / "ex15.py"
OUTPUT = Path(__file__).with_name("results.csv")
LEVELS = (1, 10, 100, 1000, 10000)
VARIANTS = ("none", "jacobi", "block_jacobi", "polynomial")


def parse_fields(line: str) -> dict[str, str]:
    return dict(field.split("=", 1) for field in line.split()[1:])


def run_level(level: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "INCREASE_REFINE_MESH": str(level),
        "SKIP_VISUALISATION": "1",
        "CG_PRECOND_BENCHMARK": "1",
        "CG_CASE_TIMEOUT_S": "30",
        "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}",
    })
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    row = {"level": str(level)}
    for line in result.stdout.splitlines():
        if line.startswith("[cgbench]"):
            row.update(parse_fields(line))
        elif line.startswith("[cgcase]"):
            fields = parse_fields(line)
            variant = fields.pop("case")
            if variant in VARIANTS:
                row.update({f"{variant}_{key}": value for key, value in fields.items()})
    return row


def main() -> None:
    rows = []
    for level in LEVELS:
        row = run_level(level)
        rows.append(row)
        print(
            f"L={level:>5} DoF={row['ndofs']:>6} "
            f"CPU={row['spsolve_cpu_s']}s "
            f"GPU-block-Jacobi={row.get('block_jacobi_total_s')}s",
            flush=True,
        )

    fieldnames = [
        "level", "ndofs", "nnz", "spsolve_cpu_s", "transfer_s", "tol", "maxiter",
    ]
    for variant in VARIANTS:
        fieldnames.extend(
            f"{variant}_{key}"
            for key in ("status", "total_s", "converged", "iters", "relres")
        )
    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()