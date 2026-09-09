# Example 33: CPU CG vs GPU CG with four preconditioners

**Question.** For the Example 33 curl–curl + mass system (lowest-order Nédélec
edge elements on a growing tensor tetrahedral mesh), how does the Conjugate
Gradient method compare **on the CPU versus the GPU**, and how do four
preconditioners behave on each device, relative to the CPU direct solver
`scipy.sparse.linalg.spsolve`?

This is the dedicated single-example device-vs-device study that backs the
Example 33 figures in the thesis. It complements the whole-suite campaign in
[`../cg_all_examples/`](../cg_all_examples/) (which measures GPU CG end-to-end
against `spsolve` across every CG-compatible example) by additionally timing
**CPU CG** so that the effect of the execution device can be isolated per
preconditioner.

---

## What is measured

For each mesh-refinement level the lowest-order Nédélec system is assembled and
the tangential boundary condition is condensed, yielding a symmetric positive
definite system `A x = b`. That exact condensed system is then solved with:

* **CPU `spsolve`** — the direct reference (also the source of `x_ref`);
* **CPU CG** and **GPU CG**, each with the same four preconditioners:

| case | preconditioner | notes |
|------|----------------|-------|
| `none`         | — (unpreconditioned)                 | baseline Krylov iteration |
| `jacobi`       | diagonal                             | cheapest, always SPD |
| `block_jacobi` | inverse of contiguous diagonal blocks | GPU-friendly (batched dense inverse) |
| `polynomial`   | symmetric truncated Neumann series   | GPU-friendly (pure SpMV) |

The `ilu0` variant of the whole-suite campaign is intentionally excluded here:
its sequential triangular solves have no device-symmetric counterpart and it is
not part of a CPU-vs-GPU comparison.

For every `(level, device, case)` the harness records the preconditioner build
time, the CG solve time, the device→host readback time (GPU only), the
convergence status, the iteration count, the 2-norm relative residual
`‖Ax−b‖/‖b‖`, and the mean/max absolute difference from the CPU `spsolve`
reference. The per-level host→device transfer time is recorded once so an
end-to-end GPU cost can be reconstructed as
`transfer + precond + solve + readback`.

---

## Files

| File | What it is |
|------|-----------|
| `benchmark.py` | The sweep harness (assembles Example 33, times CPU/GPU CG). |
| `results.csv`  | One row per mesh level; see the column layout below. |
| `sweep.log`    | Human-readable log of the most recent run. |

The three thesis figures are produced from `results.csv` by
[`../../_thesis/thesis/plot_ex33_solver_comparisons.py`](../../_thesis/thesis/plot_ex33_solver_comparisons.py):
`ex33_cg_cpu_vs_gpu` (solve time, one panel per preconditioner),
`ex33_cg_precision_vs_spsolve` (accuracy), and
`ex33_cpu_spsolve_vs_gpu_cg` (end-to-end CPU direct vs GPU CG).

---

## How to run

Requires the CUDA GPU and CuPy. Run with the repository root on `PYTHONPATH`:

```sh
PYTHONPATH=. uv run python _findings/ex33_cg_cpu_vs_gpu/benchmark.py
```

The mesh is grown internally by a geometric sequence of points-per-direction
(`START_PTS`, `PTS_GROWTH`, `MAX_PTS`), so each level adds roughly a constant
factor of degrees of freedom; each level runs in the same process and
`results.csv` is rewritten after every level, so a partial sweep is never lost.

### Environment variables

| variable | default | meaning |
|----------|--------:|---------|
| `START_PTS`        | `6`    | points per direction at the first level |
| `MAX_PTS`          | `120`  | ceiling on points per direction |
| `PTS_GROWTH`       | `1.18` | geometric growth factor per level (min +1) |
| `MAX_LEVELS`       | `40`   | global ceiling on the number of levels |
| `SPSOLVE_CAP_S`    | `120`  | stop after a level whose CPU `spsolve` exceeds this |
| `GLOBAL_BUDGET_S`  | `3600` | wall-clock cap for the whole sweep |
| `CG_CASE_TIMEOUT_S`| `120`  | per-solve wall budget (CPU and GPU) |
| `CG_MAXITER`       | `5000` | CG iteration cap |
| `CG_BLOCK_SIZE`    | `8`    | block size for `block_jacobi` |
| `CG_POLY_DEGREE`   | `3`    | degree of the `polynomial` preconditioner |
| `TOLERANCE`        | `1e-5` | CG relative tolerance |

---

## `results.csv` columns

Per row: `level, pts_per_dir, ndofs, nnz, spsolve_cpu_s, transfer_s, tol,
maxiter`, followed, for each `{device}` in `cpu, gpu` and each `{case}` in
`none, jacobi, block_jacobi, polynomial`, by:

```
{device}_{case}_precond_s {device}_{case}_solve_s {device}_{case}_readback_s
{device}_{case}_status {device}_{case}_converged {device}_{case}_iters
{device}_{case}_relres {device}_{case}_mean_abs_diff {device}_{case}_max_abs_diff
```

(`readback_s` is `0.0` for the CPU device.) The end-to-end GPU cost for a case is
`transfer_s + gpu_{case}_precond_s + gpu_{case}_solve_s + gpu_{case}_readback_s`.

---

## Caveats

* **Fair convergence.** Judge a case by `converged=1` together with `relres`; a
  fast wall-clock time with `status=maxiter`/`timeout` is a non-converged
  (unfair) result. For this operator the degree-3 `polynomial` preconditioner
  does not reach the tolerance and runs to the iteration cap on both devices;
  its points are drawn but flagged in the figures.
* **Warm-up.** The CUDA context and both CG code paths are warmed up on a tiny
  system before the sweep so the first measured level is not charged one-time
  costs.

---

## Results (latest sweep)

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB), CuPy 14.1.1. Sweep of
10 mesh levels, `665 → 117,026` degrees of freedom, stopped when CPU `spsolve`
exceeded `SPSOLVE_CAP_S = 150 s`. Full data in `results.csv`; log in `sweep.log`.

**Solve time, CPU CG vs GPU CG** (all preconditioners share a solve-time
crossover near **13,897 dofs** — below it the CPU wins on launch overhead, above
it the GPU wins). At the largest case (**117,026 dofs**):

| preconditioner | CPU CG solve | GPU CG solve | solve speedup | converges to |
|----------------|-------------:|-------------:|--------------:|-------------:|
| none         | 3.46035 s | 0.21926 s | 15.8× | 117,026 dofs |
| jacobi       | 6.31022 s | 0.18180 s | 34.7× | 117,026 dofs |
| block_jacobi | 9.78429 s | 0.22344 s | 43.8× | 117,026 dofs |
| polynomial   | — (stalls) | — (stalls) | — | ~8,261 dofs only |

**End-to-end (GPU incl. transfer + readback) vs CPU `spsolve`.** The direct
factorization cost grows much faster than the iterative solve, so every
converging GPU CG variant overtakes `spsolve` already at **4,401 dofs**. At
**117,026 dofs**, `spsolve` takes **646.53560 s** while the full GPU pipeline
takes:

| preconditioner | GPU end-to-end | speedup vs spsolve |
|----------------|---------------:|-------------------:|
| none         | 0.23790 s | ≈ 2.72×10³ |
| jacobi       | 0.20121 s | ≈ 3.21×10³ |
| block_jacobi | 0.24629 s | ≈ 2.63×10³ |

**Accuracy.** CPU and GPU curves are nearly superposed per preconditioner
(device-independent). Relative to `spsolve`, converged GPU CG stays within a
mean absolute difference of order `1e-7` (down to `~8e-8` at the largest sizes)
and a maximum absolute difference of order `1e-6` (worst case `~5.5e-6`).

**Polynomial preconditioner (degree 3) is a mixed result.** It converges only
for the smaller meshes (up to `~8,261 dofs`); from `~13,897 dofs` it hits
`maxiter=5000` (relres `~4e-4` and worse), and beyond `~70,246 dofs` it stalls
entirely (timeout, `relres` diverges). Its fixed low-degree Neumann series is
too weak for the stiffer curl–curl systems. It is reported for completeness but
excluded from the speedup summaries.
