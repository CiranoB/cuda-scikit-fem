# GMRES (GPU) vs SPSOLVE (CPU): when does the iterative GPU solver win?

**Question.** Is there a use case where solving the linear system with **GMRES on
the GPU** is faster than the direct sparse solver **`scipy.sparse.linalg.spsolve`
on the CPU**?

**Answer.** Yes — but the dimensionality of the PDE is decisive:

* In **2D** the CPU direct solver is essentially unbeatable.
* In **3D** GPU GMRES wins by **one to three orders of magnitude**.

---

## Files

| File | What it does |
|------|--------------|
| `benchmark.py`    | 2D advection-diffusion (ex50), refinement sweep. |
| `benchmark_3d.py` | 3D advection-diffusion on the unit cube, grid sweep. |
| `results_*.csv`   | Raw timings produced by the scripts. |

Run with the repository root on `PYTHONPATH` (so the local `skfem` is used):

```sh
PYTHONPATH=. uv run python _findings/gmres_gpu_vs_spsolve_cpu/benchmark_3d.py
PYTHONPATH=. uv run python _findings/gmres_gpu_vs_spsolve_cpu/benchmark.py
```

---

## Result in 3D (the winning case)

3D advection-diffusion `-κ Δu + (β·∇)u = f` on `[0,1]³`, `κ = β = 1`
(diffusion-dominated so the **non-symmetric** matrix is well conditioned),
GMRES with a **Jacobi** preconditioner, `rtol = 1e-5`, `restart = 200`.
GPU = single CUDA device (~8 GB).

| grid n | dofs | spsolve CPU | GMRES GPU (total) | speed-up | GMRES converged? |
|-------:|-----:|------------:|------------------:|---------:|:----------------:|
| 20 |   5 832 |    2.15 s |  0.17 s |   **12×**  | yes (relres 4e-13) |
| 30 |  21 952 |    2.42 s |  0.05 s |   **50×**  | yes (relres 2e-13) |
| 40 |  54 872 |   22.21 s |  0.12 s |  **192×**  | yes (relres 5e-13) |
| 50 | 110 592 |  131.53 s |  0.24 s |  **541×**  | yes (relres 4e-11) |
| 60 | 195 112 |  512.31 s |  0.39 s | **1306×**  | yes (relres 1e-08) |

"Converged" means GMRES returned `info = 0` and reached a 2-norm relative
residual well below the requested tolerance — i.e. the comparison is **fair**:
both solvers deliver an accurate solution.

### Why 3D wins
A sparse **direct** factorization in 3D suffers catastrophic fill-in
(work `~O(N²)`, factor memory `~O(N^{4/3})`), so `spsolve` blows up: 22 s at
55 k dofs, 132 s at 110 k dofs, **512 s at 195 k dofs**. A **GMRES iteration is
just one sparse
mat-vec**, which is massively parallel and ideal for the GPU; for this
well-conditioned system it converges in a fraction of a second regardless of
the problem size.

---

## Result in 2D (the negative control)

2D advection-diffusion of `ex50` (`cylinder_stokes.msh`, Peclet 30), refinement
levels 5–7. **Whenever GMRES actually converges it is *slower* than the CPU
direct solver.** The only apparent "wins" came from GMRES stopping at
`maxiter` with a poor residual (`info ≠ 0`), which is not a valid comparison.

Reasons:

1. 2D sparse direct fill-in is mild (`~O(N^1.5)`); `spsolve` solves 150 k dofs
   in ~1–3 s.
2. The FEM conditioning degrades as the mesh is refined (`κ ~ h⁻²`), so GMRES
   needs more iterations exactly when the problem grows.
3. A larger `restart` improves convergence but the orthogonalization cost
   `~O(restart²·N)` per cycle quickly dominates.

---

## Practical lessons (cupyx)

* **Do not use the cupyx ILU preconditioner here.** Its sparse *triangular
  solves* are sequential on the GPU and dominate the runtime (tens of seconds
  for 150 k dofs). A **Jacobi** preconditioner is effectively free and lets
  every GMRES iteration stay a single parallel SpMV.
* Judge convergence by the **2-norm relative residual** `‖Ax−b‖/‖b‖`, not by a
  max element-wise relative error — the latter is dominated by near-zero
  solution entries and is misleading.
* Always check the GMRES `info` flag: a wall-clock "win" with `info ≠ 0` is a
  non-converged solve, not a real speed-up.

---

## Take-away for the thesis

> GMRES on the GPU beats CPU `spsolve` for **3D** (or otherwise high-fill-in),
> **well-conditioned, non-symmetric** systems, where the direct solver's
> super-linear fill-in is the bottleneck. For **2D** problems the CPU direct
> solver remains the better choice.
