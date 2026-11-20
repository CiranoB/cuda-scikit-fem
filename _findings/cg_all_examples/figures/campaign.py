"""Loader for the repeated CG-campaign results (``run_*/results.csv``).

A ``(example, level)`` point is kept only if it exists in every run.  A GPU
variant counts as converged at a point only if it converged in *every* run;
its time is then the mean over the runs.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = (
    REPO_ROOT / "docs" / "examples" / "cg_all_examples_results"
    / "server_results" / "20261019T103612"
)

GPU_VARIANTS = ("none", "jacobi", "block_jacobi", "polynomial", "ilu0")
CPU_VARIANTS = ("none", "jacobi", "block_jacobi", "polynomial")

# Fixed-size or non-convergent examples, excluded from the preconditioner
# comparison (they have no growing mesh to compare variants on).
NON_GROWING_EXAMPLES = {15, 41, 47, 48, 49}


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


@dataclass
class Variant:
    times: list[float]  # per-run total_s of the runs that converged
    n_runs: int
    iters: list[int] = field(default_factory=list)
    relres: list[float] = field(default_factory=list)
    max_diff: list[float] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)

    @property
    def timed_out(self) -> bool:
        return bool(self.statuses) and all(s == "timeout" for s in self.statuses)

    @property
    def converged(self) -> bool:
        return len(self.times) == self.n_runs

    @property
    def mean(self) -> float:
        return statistics.fmean(self.times)

    @property
    def std(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0


@dataclass
class Point:
    example: int
    level: int
    ndofs: int
    spsolve: list[float]
    gpu: dict[str, Variant]
    cpu: dict[str, Variant]

    @property
    def spsolve_mean(self) -> float:
        return statistics.fmean(self.spsolve)

    @property
    def spsolve_std(self) -> float:
        return statistics.stdev(self.spsolve) if len(self.spsolve) > 1 else 0.0

    def converged_gpu(self) -> dict[str, Variant]:
        return {k: v for k, v in self.gpu.items() if v.converged}

    def best_gpu(self) -> tuple[str, Variant] | None:
        conv = self.converged_gpu()
        if not conv:
            return None
        name = min(conv, key=lambda k: conv[k].mean)
        return name, conv[name]

    def best_cpu(self) -> tuple[str, Variant] | None:
        conv = {k: v for k, v in self.cpu.items() if v.converged}
        if not conv:
            return None
        name = min(conv, key=lambda k: conv[k].mean)
        return name, conv[name]


def _variant(rows: list[dict], prefix: str) -> Variant:
    times, iters, relres, max_diff = [], [], [], []
    statuses = [row.get(f"{prefix}_status", "") for row in rows]
    for row in rows:
        total = _float(row.get(f"{prefix}_total_s"))
        if row.get(f"{prefix}_converged") == "1" and total is not None:
            times.append(total)
            iters.append(int(float(row.get(f"{prefix}_iters") or 0)))
            relres.append(_float(row.get(f"{prefix}_relres")) or 0.0)
            max_diff.append(_float(row.get(f"{prefix}_max_abs_diff")) or 0.0)
    return Variant(times, len(rows), iters, relres, max_diff, statuses)


def run_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("run_*/results.csv"),
                  key=lambda p: int(p.parent.name.split("_")[1]))


def load_runs(results_dir: Path) -> list[dict[tuple[int, int], dict]]:
    runs = []
    for path in run_files(results_dir):
        with path.open(newline="") as fh:
            runs.append({(int(r["example"]), int(r["refined"])): r
                         for r in csv.DictReader(fh)})
    if not runs:
        raise SystemExit(f"no run_*/results.csv under {results_dir}")
    return runs


def load_points(results_dir: Path = DEFAULT_RESULTS_DIR) -> list[Point]:
    runs = load_runs(results_dir)
    common = sorted(set.intersection(*(set(r) for r in runs)))
    points = []
    for key in common:
        rows = [r[key] for r in runs]
        sp = [_float(r["spsolve_cpu_s"]) for r in rows]
        sp = [v for v in sp if v is not None]
        points.append(Point(
            example=key[0], level=key[1], ndofs=int(rows[0]["ndofs"]),
            spsolve=sp,
            gpu={v: _variant(rows, v) for v in GPU_VARIANTS},
            cpu={v: _variant(rows, f"cpu_{v}") for v in CPU_VARIANTS},
        ))
    return points


def by_example(points: list[Point]) -> dict[int, list[Point]]:
    out: dict[int, list[Point]] = {}
    for p in points:
        out.setdefault(p.example, []).append(p)
    for pts in out.values():
        pts.sort(key=lambda p: p.ndofs)
    return out


def largest_converged(points: list[Point]) -> Point | None:
    """Highest-DoF point where at least one GPU variant converged in all runs."""
    for p in reversed(points):
        if p.spsolve and p.best_gpu() is not None:
            return p
    return None
