"""Mean / sample standard deviation of timings across repeated campaigns.

Usage: python aggregate_repeats.py <results_dir>

``<results_dir>`` must contain ``run_*/results.csv``.  Only ``(example,
refined)`` pairs present in every run are aggregated.  Writes
``<results_dir>/aggregate.csv`` and prints the largest common level per
example.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path


def load(path: Path) -> dict[tuple[int, int], dict]:
    with path.open(newline="") as fh:
        return {(int(r["example"]), int(r["refined"])): r for r in csv.DictReader(fh)}


def to_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def main() -> None:
    root = Path(sys.argv[1])
    runs = [load(p) for p in sorted(root.glob("run_*/results.csv"))]
    if not runs:
        sys.exit(f"no run_*/results.csv under {root}")

    common = sorted(set.intersection(*(set(r) for r in runs)))
    metrics = [c for c in next(iter(runs[0].values())) if c == "spsolve_cpu_s"
               or c == "transfer_s" or c.endswith("_total_s")]

    out_rows = []
    for key in common:
        for metric in metrics:
            values = [to_float(r[key].get(metric)) for r in runs]
            values = [v for v in values if v is not None]
            if len(values) < 2:
                continue
            mean = statistics.fmean(values)
            std = statistics.stdev(values)
            out_rows.append({
                "example": key[0], "refined": key[1],
                "ndofs": runs[0][key]["ndofs"], "metric": metric,
                "n": len(values), "mean_s": mean, "std_s": std,
                "cv_pct": 100 * std / mean if mean else "",
            })

    out = root / "aggregate.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    last: dict[int, int] = {}
    for ex, lvl in common:
        last[ex] = max(last.get(ex, 0), lvl)
    print(f"{len(runs)} runs; largest common level per example "
          f"(mean +- std, CV%):")
    for ex, lvl in sorted(last.items()):
        parts = [f"{r['metric'].removesuffix('_total_s').removesuffix('_cpu_s')}="
                 f"{r['mean_s']:.4g}+-{r['std_s']:.2g}({r['cv_pct']:.0f}%)"
                 for r in out_rows
                 if r["example"] == ex and r["refined"] == lvl
                 and r["metric"] != "transfer_s"]
        ndofs = next((r["ndofs"] for r in out_rows
                      if r["example"] == ex and r["refined"] == lvl), "?")
        print(f"ex{ex:02d} L{lvl} ndofs={ndofs}: " + " ".join(parts))
    print(f"\nFull table: {out}")


if __name__ == "__main__":
    main()
