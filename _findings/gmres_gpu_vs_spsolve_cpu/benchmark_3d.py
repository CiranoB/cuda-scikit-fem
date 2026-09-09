"""Benchmark: GMRES (GPU) vs SPSOLVE (CPU) on a *3D* non-symmetric problem.

Motivation
----------
The 2D advection-diffusion experiment (see ``benchmark.py`` and the README)
showed that, for two-dimensional problems, a direct sparse solver on the CPU
(``scipy.sparse.linalg.spsolve``, UMFPACK/SuperLU) is essentially unbeatable:
the fill-in grows only like ``O(N^1.5)`` while the FEM conditioning degrades
with refinement, so an honestly converged GMRES never wins.

The situation is completely different in **3D**, where the fill-in of a sparse
direct factorization explodes (work ``~O(N^2)``, factor memory ``~O(N^{4/3})``).
This is the textbook regime where a matrix-free / iterative GPU solver beats a
direct CPU solver.

Problem
-------
Advection-diffusion on the unit cube ``[0, 1]^3``::

    -kappa * laplace(u) + (beta . grad) u = f          in  (0,1)^3
                                       u  = 0           on  boundary

with a constant transport velocity ``beta``.  The advection term makes the
system matrix **non-symmetric**, so GMRES (not CG) is the appropriate Krylov
method.  ``kappa`` is chosen large enough (diffusion-dominated) that GMRES with
a cheap Jacobi preconditioner converges, while the matrix stays non-symmetric.

Run
---
    PYTHONPATH=. uv run python _findings/gmres_gpu_vs_spsolve_cpu/benchmark_3d.py

Environment variables
---------------------
START_N / END_N   first/last number of grid points per side (default 20 / 60)
STEP_N            increment of grid points per side (default 10)
KAPPA             diffusion coefficient (default 1.0)
BETA              advection speed along (1,1,1)/sqrt(3) (default 1.0)
TOLERANCE         GMRES relative tolerance (default 1e-5)
RESTART           GMRES restart / Krylov subspace size (default 200)
MAXITER           GMRES maximum outer iterations (default 8000)
"""

import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

import cupy as cp
from cupyx.scipy.sparse.linalg import gmres
from cupyx.scipy.sparse import csr_matrix as cpx_csr

from skfem import MeshTet, Basis, ElementTetP1, BilinearForm, LinearForm
from skfem import asm, condense
from skfem.helpers import grad, dot

from skfem.sparse_solvers import get_jacobi_preconditioner


KAPPA = float(os.getenv("KAPPA", "1.0"))
BETA = float(os.getenv("BETA", "1.0"))
TOLERANCE = float(os.getenv("TOLERANCE", "1e-5"))
START_N = int(os.getenv("START_N", "20"))
END_N = int(os.getenv("END_N", "60"))
STEP_N = int(os.getenv("STEP_N", "10"))
RESTART = int(os.getenv("RESTART", "200"))
MAXITER = int(os.getenv("MAXITER", "8000"))

# constant transport direction (normalised)
_BDIR = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)


@BilinearForm
def diffusion(u, v, w):
    return KAPPA * dot(grad(u), grad(v))


@BilinearForm
def advection(u, v, w):
    g = grad(u)
    return BETA * (_BDIR[0] * g[0] + _BDIR[1] * g[1] + _BDIR[2] * g[2]) * v


@LinearForm
def load(v, w):
    return 1.0 * v


def assemble(n: int):
    """Assemble the condensed 3D advection-diffusion system."""
    line = np.linspace(0.0, 1.0, n)
    mesh = MeshTet.init_tensor(line, line, line)
    basis = Basis(mesh, ElementTetP1())

    A = asm(diffusion, basis) + asm(advection, basis)
    b = asm(load, basis)

    D = basis.get_dofs()
    Acond, bcond, _, _ = condense(A, b, D=D)
    return Acond.tocsr(), bcond


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    cp.cuda.Stream.null.synchronize()
    return result, time.perf_counter() - start


def run_level(n: int):
    A, b = assemble(n)
    ndofs = A.shape[0]
    nnz = A.nnz
    bnorm = float(np.linalg.norm(b))

    # --- CPU direct reference --------------------------------------------
    x_cpu, t_spsolve_cpu = timed(spla.spsolve, A, b)

    # --- GPU GMRES (Jacobi preconditioned) -------------------------------
    def _to_gpu():
        return (cpx_csr(A, dtype=cp.float64),
                cp.asarray(b, dtype=cp.float64))

    (A_gpu, b_gpu), t_transfer = timed(_to_gpu)
    M, t_precond = timed(get_jacobi_preconditioner, A_gpu)

    def _gmres():
        return gmres(A_gpu, b_gpu, rtol=TOLERANCE, restart=RESTART,
                     maxiter=MAXITER, M=M)

    (gmres_out), t_gmres = timed(_gmres)
    x_gpu, info = gmres_out
    x_gmres, t_readback = timed(lambda: cp.asnumpy(x_gpu))

    t_gpu_total = t_transfer + t_precond + t_gmres + t_readback
    t_gpu_solve_only = t_precond + t_gmres

    relres = float(np.linalg.norm(A @ x_gmres - b) / (bnorm + 1e-30))

    del A_gpu, b_gpu, M, x_gpu
    cp.get_default_memory_pool().free_all_blocks()

    speedup_total = t_spsolve_cpu / t_gpu_total
    speedup_solve = t_spsolve_cpu / t_gpu_solve_only
    converged = info == 0 and relres <= 10 * TOLERANCE
    win = ""
    if t_gpu_total < t_spsolve_cpu:
        win = "  <== GMRES GPU WINS" + ("" if converged else " (NOT converged)")

    print(
        f"n={n:3d} ndofs={ndofs:>9d} nnz={nnz:>11d} | "
        f"spsolve_cpu={t_spsolve_cpu:9.3f}s | "
        f"[xfer={t_transfer:5.2f} prec={t_precond:5.2f} "
        f"gmres={t_gmres:8.3f} read={t_readback:4.2f}] | "
        f"gpu_total={t_gpu_total:8.3f}s (x{speedup_total:6.2f}) | "
        f"info={info:>5d} relres={relres:.1e}{win}",
        flush=True,
    )

    return {
        "n": n,
        "ndofs": ndofs,
        "nnz": nnz,
        "spsolve_cpu_s": t_spsolve_cpu,
        "transfer_s": t_transfer,
        "precond_s": t_precond,
        "gmres_s": t_gmres,
        "readback_s": t_readback,
        "gpu_total_s": t_gpu_total,
        "gpu_solve_only_s": t_gpu_solve_only,
        "speedup_total": speedup_total,
        "speedup_solve_only": speedup_solve,
        "gmres_info": info,
        "relres": relres,
        "converged": int(converged),
    }


def main():
    print(
        f"3D advection-diffusion  kappa={KAPPA} beta={BETA} tol={TOLERANCE} "
        f"restart={RESTART} maxiter={MAXITER} "
        f"n={START_N}..{END_N} step={STEP_N}",
        flush=True,
    )

    fields = [
        "n", "ndofs", "nnz",
        "spsolve_cpu_s", "transfer_s", "precond_s", "gmres_s", "readback_s",
        "gpu_total_s", "gpu_solve_only_s",
        "speedup_total", "speedup_solve_only", "gmres_info", "relres",
        "converged",
    ]
    rows = []
    for n in range(START_N, END_N + 1, STEP_N):
        try:
            rows.append(run_level(n))
        except cp.cuda.memory.OutOfMemoryError:
            print(f"n={n}: GPU out of memory, stopping.", flush=True)
            cp.get_default_memory_pool().free_all_blocks()
            break
        except MemoryError:
            print(f"n={n}: host out of memory, stopping.", flush=True)
            break

    out_csv = Path(__file__).resolve().parent / "results_3d.csv"
    with out_csv.open("w") as fh:
        fh.write(",".join(fields) + "\n")
        for row in rows:
            fh.write(",".join(f"{row[k]}" for k in fields) + "\n")
    print(f"\nSaved {out_csv}", flush=True)


if __name__ == "__main__":
    main()
