# CG-compatible examples: GPU CG (5 preconditioners) vs CPU `spsolve`

**Question.** For every documentation example whose assembled system is
compatible with the Conjugate Gradient method (symmetric positive definite),
how does GPU CG — under a range of preconditioners — compare against the CPU
direct solver `scipy.sparse.linalg.spsolve` as the mesh is refined?

This experiment generalises the single-example Example 33 study to the whole
CG-compatible subset of the example suite, and additionally sweeps five
preconditioner choices per solve.

---

## What is measured

For each example × mesh-refinement level, the **first** SPD linear solve
`A x = b` is intercepted (see below) and the following five **GPU CG** cases
are each timed end-to-end and compared against the CPU `spsolve` reference:

| case | preconditioner | notes |
|------|----------------|-------|
| `none`         | — (unpreconditioned)            | baseline Krylov iteration |
| `jacobi`       | diagonal                        | cheapest, always SPD |
| `ilu0`         | incomplete LU (cuSPARSE)        | **negative control**: sequential triangular solves are slow on the GPU |
| `block_jacobi` | inverse of contiguous diagonal blocks | GPU-friendly (batched dense inverse, one SpMV per apply) |
| `polynomial`   | symmetric truncated Neumann series | GPU-friendly (pure SpMV, no triangular solves) |

"End-to-end" for each GPU case = host→device transfer + preconditioner build +
CG solve + device→host readback. The CUDA context is warmed up before the
transfer is timed, so `transfer_s` reflects the genuine host→device copy rather
than one-time context initialisation.

For every case the harness records timing, `status`
(`converged` / `maxiter` / `timeout` / `error`), iteration count, the 2-norm
relative residual `‖Ax−b‖/‖b‖`, and the absolute difference from the CPU
`spsolve` solution.

---

## Files

| File | What it is |
|------|-----------|
| `benchmark.py` | The sweep harness. |
| `results.csv`  | One row per `(example, refine level)`; 64 columns (see below). |
| `status.csv`   | One row per example: levels completed, largest DOF count, and the reason the sweep stopped. |

The CG-compatible example list is taken from the `CG` column of
[`../solver_suitability.csv`](../solver_suitability.csv).

---

## How to run

Requires the CUDA GPU and CuPy. Run with the repository root on `PYTHONPATH`
so the local `skfem` (with the GPU solvers and preconditioners) is used:

```sh
PYTHONPATH=. uv run python _findings/cg_all_examples/benchmark.py
```

The mesh of each example is grown through the `INCREASE_REFINE_MESH`
environment variable already wired into the examples (`1` = the original mesh).
Each `(example, level)` runs in its own subprocess for isolation. `results.csv`
is rewritten after every completed run, so a partial campaign is never lost.

### Environment variables

| variable | default | meaning |
|----------|--------:|---------|
| `START_LEVEL`       | `1`    | first `INCREASE_REFINE_MESH` value |
| `MAX_LEVEL`         | `20`   | global refinement ceiling |
| `PER_RUN_TIMEOUT_S` | `900`  | wall-clock cap per subprocess |
| `CG_CASE_TIMEOUT_S` | `30`   | per-preconditioner solve budget (bounds the `ilu0` negative control) |
| `CG_MAXITER`        | `5000` | CG iteration cap |
| `CG_BLOCK_SIZE`     | `8`    | block size for `block_jacobi` |
| `CG_POLY_DEGREE`    | `3`    | degree of the `polynomial` preconditioner |
| `TOLERANCE`         | `1e-5` | CG relative tolerance |
| `EXAMPLES`          | all CG | comma-separated override, e.g. `EXAMPLES=33,6,7` |

### Per-example stopping rule

Refinement increases by one level until any of: GPU/host out-of-memory, a
non-zero subprocess exit, the per-run wall-clock cap (typically hit because the
CPU `spsolve` baseline has become very slow — the regime we want to bound), or
the global `MAX_LEVEL`. The stop reason is recorded in `status.csv`.

---

## `results.csv` columns

Per row: `example, file, refined, ndofs, nnz, spsolve_cpu_s, transfer_s, tol,
maxiter`, followed, for each `{case}` in
`none, jacobi, ilu0, block_jacobi, polynomial`, by:

```
{case}_status {case}_precond_s {case}_solve_s {case}_readback_s {case}_total_s
{case}_converged {case}_iters {case}_relres {case}_max_abs_diff
{case}_mean_abs_diff {case}_speedup_total
```

where `{case}_total_s = transfer_s + precond_s + solve_s + readback_s` and
`{case}_speedup_total = spsolve_cpu_s / {case}_total_s`.

---

## Caveats

* **First-solve measurement.** Only the first SPD `A x = b` solve per process is
  benchmarked, so time-stepping / nonlinear examples are measured on their
  first solve. For a few examples the first linear solve is a post-processing
  projection (mass-matrix) solve rather than the headline stiffness system;
  such systems are still SPD and CG-amenable, so they are reported.
* **`ilu0` is a deliberate negative control.** cuSPARSE incomplete-LU
  triangular solves are sequential and dominate GPU runtime; expect it to be
  slow and to frequently hit `status=timeout`. It is included to document that
  triangular-solve preconditioners do not suit this GPU CG pipeline.
* **Fair convergence.** Judge a case by `status=converged` together with
  `relres`; a fast wall-clock time with `status=maxiter`/`timeout` is a
  non-converged (unfair) result, not a real speed-up.
* Examples that never perform a default linear solve (e.g. pure eigenvalue
  problems, or those passing an explicit iterative solver) emit no benchmark
  line and are recorded in `status.csv` with reason `no_linear_solve`.
