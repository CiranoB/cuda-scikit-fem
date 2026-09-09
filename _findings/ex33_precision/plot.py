"""Figure for the arithmetic-precision study of Example 33 GPU CG.

Reads ``results.csv`` and ``tol_floor.csv`` produced by ``benchmark.py`` and
writes ``ex33_precision_fp64_vs_fp32.{pdf,png}`` into the thesis image folder.

Three panels:
  (a) GPU CG solve time vs degrees of freedom, double vs single precision;
  (b) attained (true, float64) relative residual vs degrees of freedom at the
      thesis tolerance of 1e-5, showing that single precision drifts away from
      the target as the system grows while double precision holds it;
  (c) tolerance-floor sweep at a fixed mesh: the attained relative residual as
      the requested tolerance is tightened, showing that single precision
      stagnates at its round-off floor while double precision keeps converging.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

# All figure fonts scaled up 30% relative to matplotlib's defaults.
plt.rcParams["font.size"] *= 1.3

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
IMG_DIR = REPO_ROOT / "_thesis" / "thesis" / "img"
RESULTS_CSV = HERE / "results.csv"
FLOOR_CSV = HERE / "tol_floor.csv"

FP64_COLOR = "#1f77b4"
FP32_COLOR = "#d62728"
TOL = 1e-5


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, name):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(IMG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    rows = load(RESULTS_CSV)
    ndofs = [_f(r["ndofs"]) for r in rows]

    # Figure 1: solve time -------------------------------------------------
    fig, ax_t = plt.subplots(figsize=(6, 4.4))
    series = [
        ("fp64", "none", FP64_COLOR, "--", "o", "double, no precond."),
        ("fp64", "jacobi", FP64_COLOR, "-", "s", "double, Jacobi"),
        ("fp32", "none", FP32_COLOR, "--", "o", "single, no precond."),
        ("fp32", "jacobi", FP32_COLOR, "-", "s", "single, Jacobi"),
    ]
    for prec, case, color, ls, marker, label in series:
        y = [_f(r[f"{prec}_{case}_solve_s"]) for r in rows]
        ax_t.plot(ndofs, y, ls=ls, marker=marker, color=color, label=label)
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_xlabel("degrees of freedom")
    ax_t.set_ylabel("GPU CG solve time [s]")
    ax_t.set_title("Solve time: double vs single precision")
    ax_t.grid(True, which="both", ls=":", alpha=0.5)
    ax_t.legend(fontsize=11.7)
    fig.tight_layout()
    save_figure(fig, "ex33_precision_solve_time")

    # Figure 2: attained true residual vs size at tol=1e-5 -----------------
    fig, ax_r = plt.subplots(figsize=(6, 4.4))
    for prec, color, marker, label in (
        ("fp64", FP64_COLOR, "s", "double, Jacobi"),
        ("fp32", FP32_COLOR, "s", "single, Jacobi"),
    ):
        y = [_f(r[f"{prec}_jacobi_relres"]) for r in rows]
        ax_r.plot(ndofs, y, ls="-", marker=marker, color=color, label=label)
    ax_r.axhline(TOL, color="black", ls=":", lw=1.2,
                 label=f"requested tol = {TOL:g}")
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.set_xlabel("degrees of freedom")
    ax_r.set_ylabel(r"attained residual $\|Ax-b\|/\|b\|$")
    ax_r.set_title("Attained accuracy at tol = 1e-5")
    ax_r.grid(True, which="both", ls=":", alpha=0.5)
    ax_r.legend(fontsize=11.7)
    fig.tight_layout()
    save_figure(fig, "ex33_precision_accuracy")

    # Figure 3: tolerance floor at a fixed mesh ----------------------------
    floor = load(FLOOR_CSV)
    fixed_dofs = int(_f(floor[0]["ndofs"])) if floor else 0
    fig, ax_f = plt.subplots(figsize=(6, 4.4))
    for prec, color, marker, label in (
        ("fp64", FP64_COLOR, "s", "double"),
        ("fp32", FP32_COLOR, "s", "single"),
    ):
        pr = [r for r in floor if r["precision"] == prec]
        tol = [_f(r["requested_tol"]) for r in pr]
        rr = [_f(r["attained_relres"]) for r in pr]
        ax_f.plot(tol, rr, ls="-", marker=marker, color=color, label=label)
    lo = min(_f(r["requested_tol"]) for r in floor)
    hi = max(_f(r["requested_tol"]) for r in floor)
    ax_f.plot([lo, hi], [lo, hi], color="black", ls=":", lw=1.2,
              label="ideal (attained = requested)")
    ax_f.set_xscale("log")
    ax_f.set_yscale("log")
    ax_f.invert_xaxis()
    ax_f.set_xlabel("requested tolerance")
    ax_f.set_ylabel(r"attained residual $\|Ax-b\|/\|b\|$")
    ax_f.set_title(f"Tolerance floor at {fixed_dofs:,} dofs")
    ax_f.grid(True, which="both", ls=":", alpha=0.5)
    ax_f.legend(fontsize=11.7)
    fig.tight_layout()
    save_figure(fig, "ex33_precision_tol_floor")

    print("wrote ex33_precision_solve_time / _accuracy / _tol_floor .{pdf,png}")

    # brief numeric summary for the thesis prose
    print(f"sizes: {[int(d) for d in ndofs]}")
    for r in rows:
        d = int(_f(r["ndofs"]))
        s64 = _f(r["fp64_jacobi_solve_s"])
        s32 = _f(r["fp32_jacobi_solve_s"])
        rr64 = _f(r["fp64_jacobi_relres"])
        rr32 = _f(r["fp32_jacobi_relres"])
        print(f"  dofs={d:>7} jac_solve fp64={s64:.4f} fp32={s32:.4f} "
              f"speedup={s64/s32:.2f}x  relres fp64={rr64:.2e} fp32={rr32:.2e}")


if __name__ == "__main__":
    main()
