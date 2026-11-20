"""Per-example scaling figures for the repeated CG campaign.

For every example with a growing mesh, plots the CPU ``spsolve`` time and the
end-to-end GPU CG time of each preconditioner against the number of DoF.
Solid lines are GPU CG and dashed lines with hollow markers are CPU CG (SciPy,
same precision), both per preconditioner.  Markers are means over the campaign
rounds and error bars are the sample standard deviation.  A variant is drawn at
a mesh level only if it converged in every round.  ILU(0) is the negative
control (no CPU counterpart) and is not drawn.

Usage:
    uv run python _findings/cg_all_examples/figures/plot_scaling.py \
        [results_dir] [output_dir]

Writes ``exNN_cg_scaling.{pdf,png}`` and ``cg_all_examples_scaling.{pdf,png}``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from campaign import DEFAULT_RESULTS_DIR, by_example, load_points

plt.rcParams["font.size"] *= 1.3

# Example 15 only reaches 899 DoF here; its larger-mesh data comes from the
# dedicated sweep in _findings/ex15_scaling.
EXCLUDED_EXAMPLES = {15}

# (example, level, variant) CPU CG points reported as not converged: one of the
# 10 runs needed ~8x the iterations (std > mean), so the point is not drawn.
CPU_NOT_CONVERGED = {(3, 2, "polynomial")}

STYLES = {
    "spsolve": dict(color="black", marker="o", label="CPU spsolve"),
    "none": dict(color="tab:blue", marker="s", label="No precond."),
    "jacobi": dict(color="tab:orange", marker="^", label="Jacobi"),
    "block_jacobi": dict(color="tab:green", marker="D", label="Block-Jacobi"),
    "polynomial": dict(color="tab:red", marker="v", label="Polynomial"),
}
DRAWN_VARIANTS = ("none", "jacobi", "block_jacobi", "polynomial")


def series(points, variant, device="gpu"):
    xs, ys, es = [], [], []
    for p in points:
        v = (p.gpu if device == "gpu" else p.cpu)[variant]
        if device == "cpu" and (p.example, p.level, variant) in CPU_NOT_CONVERGED:
            continue
        if v.converged and v.mean > 0:
            xs.append(p.ndofs)
            ys.append(v.mean)
            es.append(v.std)
    return xs, ys, es


def draw(ax, points) -> None:
    s = STYLES["spsolve"]
    ax.errorbar([p.ndofs for p in points], [p.spsolve_mean for p in points],
                yerr=[p.spsolve_std for p in points], color=s["color"],
                marker=s["marker"], markersize=3.5, linewidth=2.0,
                capsize=2, elinewidth=0.8, zorder=5)
    for variant in DRAWN_VARIANTS:
        s = STYLES[variant]
        xs, ys, es = series(points, variant, "gpu")
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=s["color"], marker=s["marker"],
                        markersize=3.0, linewidth=1.4, capsize=2,
                        elinewidth=0.8)
        xs, ys, es = series(points, variant, "cpu")
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=s["color"], marker=s["marker"],
                        markersize=3.0, markerfacecolor="none", linewidth=1.2,
                        linestyle="--", capsize=2, elinewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)


def legend_handles(points=None) -> list[Line2D]:
    def drawn(device):
        return {v for v in DRAWN_VARIANTS if series(points, v, device)[0]}

    if points is None:
        gpu = cpu = set(DRAWN_VARIANTS)
    else:
        gpu, cpu = drawn("gpu"), drawn("cpu")
    handles = [Line2D([0], [0], color="black", marker="o", linewidth=2.0,
                      markersize=5, label="CPU spsolve")]
    for v in DRAWN_VARIANTS:
        if v in gpu or v in cpu:
            s = STYLES[v]
            handles.append(Line2D([0], [0], color=s["color"],
                                  marker=s["marker"], linewidth=2.0,
                                  markersize=5, label=s["label"]))
    handles.append(Line2D([0], [0], color="gray", linewidth=2.0,
                          label="GPU CG (solid)"))
    handles.append(Line2D([0], [0], color="gray", linewidth=2.0,
                          linestyle="--", label="CPU CG (dashed)"))
    return handles


def make_individual(example: int, points, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    draw(ax, points)
    ax.set_xlabel("Degrees of freedom")
    ax.set_ylabel("End-to-end time (s)")
    fig.legend(handles=legend_handles(points), loc="outside lower center",
               ncol=3, fontsize=10)
    stem = out / f"ex{example:02d}_cg_scaling"
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=200)
    plt.close(fig)


def make_grid(groups: dict[int, list], out: Path) -> None:
    examples = sorted(groups)
    ncol = 4
    nrow = math.ceil(len(examples) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.3, nrow * 2.5),
                             constrained_layout=True)
    flat = axes.flatten()
    for ax, ex in zip(flat, examples):
        draw(ax, groups[ex])
        ax.set_title(f"ex{ex:02d}", fontsize=13)
        ax.tick_params(labelsize=9.1)
    for ax in flat[len(examples):]:
        ax.axis("off")
    fig.supxlabel("Degrees of freedom", fontsize=14.3)
    fig.supylabel("End-to-end time (s)", fontsize=14.3)
    fig.legend(handles=legend_handles(), loc="outside upper center", ncol=4,
               fontsize=11.7)
    stem = out / "cg_all_examples_scaling"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "img"
    out.mkdir(parents=True, exist_ok=True)
    groups = {ex: pts for ex, pts in by_example(load_points(results_dir)).items()
              if ex not in EXCLUDED_EXAMPLES}
    for ex, pts in groups.items():
        make_individual(ex, pts, out)
    make_grid(groups, out)
    print(f"wrote {len(groups)} per-example figures and the grid to {out}")


if __name__ == "__main__":
    main()
