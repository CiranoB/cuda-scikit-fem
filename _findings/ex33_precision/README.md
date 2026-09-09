# Example 33: does GPU CG arithmetic precision matter?

**Question.** Every other experiment in the thesis runs in double precision
(`float64`). For the Example 33 SPD system, what changes if the **GPU**
Conjugate Gradient path is run in **single** (`float32`) or **half**
(`float16`) precision instead — in speed, in convergence, and in accuracy?

This study backs the thesis subsection *"Arithmetic precision: double versus
single"* (`\label{sec:precision}`).

---

## What is measured

For a growing tensor tetrahedral mesh the lowest-order Nédélec system is
assembled and the tangential boundary condition condensed, yielding a symmetric
positive definite `A x = b`. The double-precision CPU `spsolve` solution is the
reference `x_ref`. For each mesh size and each precision the harness runs GPU CG
(unpreconditioned and Jacobi-preconditioned) and records the transfer / build /
solve / readback times, the iteration count and status, and — crucially — the
relative residual `‖Ax−b‖/‖b‖` **recomputed in `float64`** against the original
operator, plus the mean/max absolute difference from `x_ref`. Recomputing the
residual in double precision is what makes the accuracy numbers honest: the
low-precision solver's own residual estimate is optimistic.

A second probe (`tol_floor.csv`) fixes one mesh and sweeps the requested CG
tolerance to expose each precision's accuracy floor.

---

## Files

| File | What it is |
|------|-----------|
| `benchmark.py`  | The sweep + tolerance-floor harness. |
| `results.csv`   | One row per mesh size; per `{precision}_{case}` timings/accuracy. |
| `tol_floor.csv` | Tolerance sweep at a fixed mesh (fp64 vs fp32, Jacobi). |
| `sweep.log`     | Human-readable log of the most recent run. |
| `plot.py`       | Builds the three thesis figures from the two CSVs. |

The three figures `ex33_precision_solve_time`, `ex33_precision_accuracy`, and
`ex33_precision_tol_floor` (`.pdf`/`.png`) are written into
`../../_thesis/thesis/img/` by `plot.py`.

---

## How to run

Requires the CUDA GPU and CuPy. Run with the repository root on `PYTHONPATH`:

```sh
PYTHONPATH=. uv run python _findings/ex33_precision/benchmark.py
uv run python _findings/ex33_precision/plot.py
```

### Environment variables

| variable | default | meaning |
|----------|--------:|---------|
| `PRECISION_PTS` | `10,12,14,16,19,23` | points/direction for the size sweep |
| `FLOOR_PTS`     | `16`   | mesh for the tolerance-floor probe |
| `CG_MAXITER`    | `5000` | CG iteration cap |
| `TOLERANCE`     | `1e-5` | requested CG tolerance for the size sweep |
| `CASE_TIMEOUT_S`| `120`  | per-solve wall budget |

---

## Results (latest sweep)

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB), CuPy 14.1.1. Size sweep
`4,401 → 70,246` dofs; tolerance-floor probe at `21,645` dofs.

**Half precision is not available.** `scipy.sparse` (and the cuSPARSE-backed
`cupyx` CSR) refuse to store a `float16` matrix
(`ValueError: scipy.sparse does not support dtype float16`). So half precision
and anything below it are out; only `float64` vs `float32` is meaningful.

**Single precision is modestly faster.** Jacobi-preconditioned GPU CG solve
time, `float64 → float32`:

| dofs | fp64 solve [s] | fp32 solve [s] | speedup |
|------|---------------:|---------------:|--------:|
| 4,401  | 0.0356 | 0.0332 | 1.07× |
| 8,261  | 0.0440 | 0.0340 | 1.29× |
| 13,897 | 0.0630 | 0.0546 | 1.15× |
| 21,645 | 0.0784 | 0.0624 | 1.26× |
| 37,962 | 0.0979 | 0.0717 | 1.37× |
| 70,246 | 0.1279 | 0.0903 | 1.42× |

The speedup grows with size but stays far below the GPU's nominal FP32:FP64
core ratio, confirming the solve is **memory-bandwidth bound** (SpMV + vector
reductions): halving the word length, not the arithmetic rate, is what helps.

**Single precision has a residual accuracy floor.** True (float64-recomputed)
residual at the requested tolerance `1e-5` (Jacobi):

| dofs | fp64 relres | fp32 relres |
|------|------------:|------------:|
| 4,401  | 9.9e-6 | 3.3e-5 |
| 21,645 | 9.1e-6 | 1.1e-4 |
| 70,246 | 9.0e-6 | 3.0e-4 |

fp64 holds `1e-5`; fp32 misses it and **worsens with problem size**. The
tolerance-floor probe (21,645 dofs) makes the mechanism explicit: fp64 follows
the requested tolerance down to `9.8e-11` at `1e-10`, while fp32 stagnates near
`1.3e-4` for every tolerance `≤ 1e-4` and merely wastes iterations (353 at
`1e-5` → 730 at `1e-10`, no improvement). This is the classic finite-precision
CG effect: the recursively updated residual drifts below the true residual and
the solver stops early.

**But the solution error stayed small.** Max abs diff from `spsolve` was
`~2e-6` for *both* precisions, because the `+E` mass term makes this operator
well conditioned, so a larger residual maps to only a small solution error. The
cushion would vanish for stiffer / ill-conditioned operators.

**Bottom line.** Double precision is the safe default: single precision buys at
most ~1.4× and cannot reach tolerances tighter than ~`1e-4`, so it is worth it
only when a residual of that order is acceptable and bandwidth is the binding
constraint; half precision and below are unusable in this sparse toolchain.
