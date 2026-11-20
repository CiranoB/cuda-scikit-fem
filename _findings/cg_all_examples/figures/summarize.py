"""Numbers and LaTeX table for the paper, from the repeated CG campaign.

Usage:
    uv run python _findings/cg_all_examples/figures/summarize.py [results_dir]

Prints (1) the Table of highest evaluated configurations, (2) headline counts,
(3) the preconditioner comparison, (4) round-to-round statistics, and (5) the
reason each example's sweep stopped.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

from campaign import (
    DEFAULT_RESULTS_DIR, GPU_VARIANTS, NON_GROWING_EXAMPLES, REPO_ROOT,
    by_example, load_points, load_runs, largest_converged,
)

EX15_DEDICATED = REPO_ROOT / "_findings" / "ex15_scaling" / "results.csv"
LABEL = {"none": "None", "jacobi": "Jacobi", "block_jacobi": "Block-Jacobi",
         "polynomial": "Polynomial", "ilu0": "ILU(0)"}


def tex_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def fmt_time(mean: float, std: float | None, floor: bool = True) -> str:
    if floor and mean < 0.01:
        return "$<0.01$"
    if mean < 0.001:
        return "$<0.001$"
    digits = 3 if mean < 1 else 2
    text = f"{mean:.{digits}f}"
    if std is not None:
        text += f"$\\pm${std:.{digits}f}"
    return text


def fmt_speedup(x: float) -> str:
    if x >= 1000:
        return f"${tex_int(round(x))}\\times$"
    if x >= 100:
        return f"${x:.0f}\\times$"
    if x >= 10:
        return f"${x:.1f}\\times$"
    return f"${x:.2f}\\times$" if x < 1 else f"${x:.1f}\\times$"


def ex15_dedicated_row() -> str:
    """Example 15 reaches 8,999 DoF only in its dedicated (single-run) sweep."""
    with EX15_DEDICATED.open(newline="") as fh:
        rows = {int(r["level"]): r for r in csv.DictReader(fh)}
    r = rows[1000]
    best = min(
        (v for v in ("none", "jacobi", "block_jacobi", "polynomial")
         if r[f"{v}_converged"] == "1"),
        key=lambda v: float(r[f"{v}_total_s"]))
    sp, gpu = float(r["spsolve_cpu_s"]), float(r[f"{best}_total_s"])
    return (f"ex15$^{{\\ddagger}}$ & {tex_int(int(r['ndofs']))} & "
            f"{fmt_time(sp, None)} & {LABEL[best]} & {gpu:.3f} & "
            f"\\multicolumn{{2}}{{c}}{{n/a}} & {fmt_speedup(sp / gpu)} & n/a \\\\")


def table(groups: dict[int, list]) -> list[str]:
    lines = []
    for ex in sorted(groups):
        if ex == 15:
            lines.append(ex15_dedicated_row())
            continue
        p = largest_converged(groups[ex])
        if p is None:
            last = groups[ex][-1]
            lines.append(
                f"ex{ex:02d} & {tex_int(last.ndofs)} & "
                f"{fmt_time(last.spsolve_mean, None)} & "
                f"\\multicolumn{{6}}{{c}}{{did not converge$^{{\\dagger}}$}} \\\\")
            continue
        name, v = p.best_gpu()
        cpu = p.best_cpu()
        if cpu is None and any(v.timed_out for v in p.cpu.values()):
            budget = 100.0  # CG_CPU_CASE_TIMEOUT_S
            cpu_cells = "--- & $>100$"
            ratio = f"$>{budget / v.mean:.1f}\\times$"
        elif cpu is None:
            cpu_cells = "\\multicolumn{2}{c}{n/c}"
            ratio = "n/c"
        else:
            cname, cv = cpu
            cpu_cells = f"{LABEL[cname]} & {fmt_time(cv.mean, cv.std, floor=False)}"
            ratio = fmt_speedup(cv.mean / v.mean)
        lines.append(
            f"ex{ex:02d} & {tex_int(p.ndofs)} & "
            f"{fmt_time(p.spsolve_mean, p.spsolve_std)} & {LABEL[name]} & "
            f"{fmt_time(v.mean, v.std, floor=False)} & {cpu_cells} & "
            f"{fmt_speedup(p.spsolve_mean / v.mean)} & {ratio} \\\\")
    return lines


def headline(groups: dict[int, list]) -> None:
    speed = {}
    for ex, pts in groups.items():
        p = largest_converged(pts)
        if p is not None and ex != 15:
            speed[ex] = p.spsolve_mean / p.best_gpu()[1].mean
    ex15 = ex15_dedicated_row()
    print(f"\nexamples with a converged GPU variant (excluding dedicated ex15): "
          f"{len(speed)}; with ex15: {len(speed) + 1}")
    faster = {e: s for e, s in speed.items() if s > 1}
    print(f"faster than spsolve: {len(faster)} (+ex15 slower); "
          f"min {min(faster.values()):.2f} (ex{min(faster, key=faster.get)}), "
          f"max {max(faster.values()):.0f} (ex{max(faster, key=faster.get)})")
    print("slower:", {f'ex{e}': round(s, 3) for e, s in speed.items() if s <= 1})
    over = {}
    for ex, pts in groups.items():
        sp = max(p.spsolve_mean for p in pts)
        if sp > 300:
            over[ex] = round(sp, 1)
    print("examples whose spsolve exceeded 300 s at some level:", over)
    print("no-GPU-converged examples:",
          [e for e, pts in groups.items() if largest_converged(pts) is None])
    top = sorted(((largest_converged(p).ndofs, e) for e, p in groups.items()
                  if largest_converged(p)), reverse=True)[:4]
    print("largest converged DoF:", top)
    print("ex15 row:", ex15)


def preconditioner_stats(groups: dict[int, list]) -> None:
    cfg = [p for ex, pts in groups.items() if ex not in NON_GROWING_EXAMPLES
           for p in pts]
    fastest, conv = Counter(), Counter()
    for p in cfg:
        best = p.best_gpu()
        if best:
            fastest[best[0]] += 1
        for k, v in p.gpu.items():
            conv[k] += v.converged
    n = len(cfg)
    print(f"\n{n} growing-mesh configurations")
    for k in GPU_VARIANTS:
        print(f"  {LABEL[k]:13s} fastest {fastest[k]:3d}/{n}  "
              f"converged {conv[k]:3d}/{n} ({100 * conv[k] / n:.1f}%)  "
              f"not converged {n - conv[k]} ({100 * (n - conv[k]) / n:.1f}%)")


def repeat_stats(results_dir: Path, points: list) -> None:
    runs = load_runs(results_dir)
    n_runs = len(runs)
    print(f"\n{n_runs} rounds; {len(points)} common (example, level) points")

    series = []  # (kind, mean, per-run times)
    for p in points:
        if len(p.spsolve) == n_runs:
            series.append(("spsolve", p.spsolve_mean, p.spsolve))
        for k, v in p.converged_gpu().items():
            series.append((f"gpu_{k}", v.mean, v.times))

    def cv(vals):
        m = statistics.fmean(vals)
        return 100 * statistics.stdev(vals) / m if m else 0.0

    for label, sel in [("all points", lambda s: True),
                       ("mean >= 1 s", lambda s: s[1] >= 1.0)]:
        sub = [s for s in series if sel(s)]
        for kind, kf in [("spsolve", lambda s: s[0] == "spsolve"),
                         ("GPU CG", lambda s: s[0] != "spsolve")]:
            cvs = sorted(cv(s[2]) for s in sub if kf(s))
            if not cvs:
                continue
            q = lambda f: cvs[min(len(cvs) - 1, int(f * len(cvs)))]
            print(f"  CV across rounds [{label}, {kind}]: n={len(cvs)} "
                  f"median {q(.5):.2f}%  p90 {q(.9):.2f}%  p99 {q(.99):.2f}%  "
                  f"max {cvs[-1]:.1f}%  share<=5% {100 * sum(c <= 5 for c in cvs) / len(cvs):.1f}%")

    print("  per-round |deviation from the across-round mean| "
          "(points with mean >= 1 s):")
    print("  round | spsolve median / max [%] | GPU CG median / max [%] | wall time [h]")
    log_times = []
    for i in range(n_runs):
        log = results_dir / f"run_{i + 1}" / "campaign.log"
        txt = log.read_text() if log.exists() else ""
        key = "Done in "
        t = float(txt.split(key)[-1].split("s.")[0]) / 3600 if key in txt else float("nan")
        log_times.append(t)
    for i in range(n_runs):
        dev = {"spsolve": [], "gpu": []}
        for p in points:
            if len(p.spsolve) == n_runs and p.spsolve_mean >= 1.0:
                dev["spsolve"].append(abs(p.spsolve[i] / p.spsolve_mean - 1) * 100)
            for v in p.converged_gpu().values():
                if v.mean >= 1.0:
                    dev["gpu"].append(abs(v.times[i] / v.mean - 1) * 100)
        med = lambda l: statistics.median(l) if l else float("nan")
        mx = lambda l: max(l) if l else float("nan")
        print(f"  {i + 1:5d} | {med(dev['spsolve']):6.2f} / {mx(dev['spsolve']):6.2f}"
              f" | {med(dev['gpu']):6.2f} / {mx(dev['gpu']):6.2f} | {log_times[i]:.2f}")

    # Determinism: convergence flags and iteration counts across rounds.
    flag_mismatch, iter_mismatch, total = 0, 0, 0
    for p in points:
        for v in p.gpu.values():
            total += 1
            if 0 < len(v.times) < n_runs:
                flag_mismatch += 1
            if len(set(v.iters)) > 1:
                iter_mismatch += 1
    print(f"  GPU variant/point pairs: {total}; convergence differs between "
          f"rounds: {flag_mismatch}; iteration count differs: {iter_mismatch}")
    sp_means = [p.spsolve_mean for p in points if p.spsolve_mean >= 1.0]
    print(f"  spsolve points >= 1 s: {len(sp_means)}")


def accuracy_stats(points: list, groups: dict[int, list]) -> None:
    """True residual ||Ax-b||/||b|| (recomputed in float64) of converged runs."""
    rel, diff = [], []
    for p in points:
        for v in p.converged_gpu().values():
            rel.extend(v.relres)
            diff.extend(v.max_diff)
    rel.sort()
    q = lambda l, f: l[min(len(l) - 1, int(f * len(l)))]
    print(f"\nGPU runs flagged converged: {len(rel)}; true relres <= 1e-5: "
          f"{100 * sum(r <= 1e-5 for r in rel) / len(rel):.1f}%  <= 1e-4: "
          f"{100 * sum(r <= 1e-4 for r in rel) / len(rel):.1f}%  <= 1e-2: "
          f"{100 * sum(r <= 1e-2 for r in rel) / len(rel):.1f}%")
    print(f"  relres median {q(rel, .5):.2e} p90 {q(rel, .9):.2e} max {rel[-1]:.2e}")
    diff.sort()
    print(f"  max|x-x_spsolve| median {q(diff, .5):.2e} p90 {q(diff, .9):.2e} "
          f"max {diff[-1]:.2e}")
    print("  at the table configuration (best variant): "
          "example relres / max_abs_diff (worst round)")
    for ex in sorted(groups):
        p = largest_converged(groups[ex])
        if p is None:
            continue
        name, v = p.best_gpu()
        print(f"    ex{ex:02d} {name:12s} {max(v.relres):.2e} {max(v.max_diff):.2e}")


def cpu_vs_gpu_stats(groups: dict[int, list]) -> None:
    """Best GPU CG against best CPU CG (both float32, same convergence rule)."""
    print("\n== CPU CG vs GPU CG ==")
    ex_rows = []
    for ex, pts in sorted(groups.items()):
        p = largest_converged(pts)
        if p is None or ex == 15:
            continue
        cpu = p.best_cpu()
        g = p.best_gpu()[1].mean
        if cpu is None:
            ex_rows.append((ex, p.ndofs, None, None, None))
        else:
            ex_rows.append((ex, p.ndofs, cpu[0], cpu[1].mean,
                            cpu[1].mean / g))
    have = [r for r in ex_rows if r[2]]
    print(f"table configs with a converged CPU CG: {len(have)}/{len(ex_rows)}; "
          f"no CPU CG: {[r[0] for r in ex_rows if not r[2]]}")
    print(f"  GPU faster than CPU CG: {[r[0] for r in have if r[4] > 1]}")
    print(f"  CPU CG faster: {[(r[0], round(r[4], 2)) for r in have if r[4] <= 1]}")
    big = [r for r in have if r[4] > 1]
    print(f"  GPU/CPU-CG speedup among GPU-faster: min {min(r[4] for r in big):.2f}"
          f" max {max(r[4] for r in big):.1f}; "
          f"median {statistics.median(r[4] for r in big):.1f}")
    print("  split of the speedup over spsolve (algorithmic spsolve->CPU CG, "
          "hardware CPU CG->GPU CG):")
    for ex, dof, cname, ct, ratio in have:
        p = largest_converged(groups[ex])
        algo = p.spsolve_mean / ct
        print(f"    ex{ex:02d} DoF {dof:>8} algo {algo:9.1f}x hw {ratio:6.2f}x "
              f"total {algo * ratio:9.1f}x")

    cfg = [p for ex, pts in groups.items() if ex not in NON_GROWING_EXAMPLES
           for p in pts]
    pairs = [(p.ndofs, p.best_cpu()[1].mean / p.best_gpu()[1].mean)
             for p in cfg if p.best_cpu() and p.best_gpu()]
    print(f"\nconfigs with both converged: {len(pairs)}/{len(cfg)}; "
          f"GPU faster in {sum(r > 1 for _, r in pairs)} "
          f"({100 * sum(r > 1 for _, r in pairs) / len(pairs):.0f}%)")
    for lo, hi in [(0, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 1e9)]:
        sub = [r for d, r in pairs if lo <= d < hi]
        if sub:
            print(f"  DoF [{lo:g},{hi:g}): n={len(sub)} median {statistics.median(sub):.2f}x "
                  f"GPU faster {100 * sum(r > 1 for r in sub) / len(sub):.0f}%")
    cpu_conv = Counter()
    cpu_stat = Counter()
    for p in cfg:
        for k, v in p.cpu.items():
            cpu_conv[k] += v.converged
    print("CPU CG converged (all rounds) of", len(cfg), dict(cpu_conv))
    for lo in (1e5,):
        same = {}
        for p in cfg:
            if p.ndofs < lo:
                continue
            for k in ("none", "jacobi", "block_jacobi", "polynomial"):
                if p.gpu[k].converged and p.cpu[k].converged:
                    same.setdefault(k, []).append(p.cpu[k].mean / p.gpu[k].mean)
        for k, v in same.items():
            print(f"  same preconditioner, DoF>={lo:g}, {k}: n={len(v)} "
                  f"median {statistics.median(v):.1f}x min {min(v):.2f}x max {max(v):.1f}x")
    # CPU CG vs spsolve crossover
    over = [(p.ndofs, p.spsolve_mean / p.best_cpu()[1].mean) for p in cfg
            if p.best_cpu()]
    print(f"CPU CG faster than spsolve in {sum(r > 1 for _, r in over)}/{len(over)}")


def stop_reasons(results_dir: Path) -> None:
    print("\nsweep stop reasons (round 1):")
    with (results_dir / "run_1" / "status.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            print(f"  ex{int(r['example']):02d}: {r['levels_done']:>3} levels, "
                  f"max {r['max_ndofs']:>8} DoF, {r['reason']}")


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    points = load_points(results_dir)
    groups = by_example(points)
    print("% --- Table rows ---")
    print("\n".join(table(groups)))
    headline(groups)
    preconditioner_stats(groups)
    repeat_stats(results_dir, points)
    accuracy_stats(points, groups)
    cpu_vs_gpu_stats(groups)
    stop_reasons(results_dir)


if __name__ == "__main__":
    main()
