"""Benchmark: GMRES (GPU) vs SPSOLVE (CPU) on a non-symmetric problem.

Goal
----
Find a use case where solving the linear system with GMRES on the GPU is
faster than the direct sparse solver ``scipy.sparse.linalg.spsolve`` on the
CPU.

The advection-diffusion problem of ``docs/examples/ex50.py`` is used because
its system matrix is non-symmetric (advection term), which is precisely the
situation where GMRES is the appropriate iterative method (CG would not be
applicable).  The mesh is refined progressively to grow the problem size and
expose the crossover point where the GPU iterative solver wins.

The same building blocks used by the library (``skfem.sparse_solvers``) are
reused so that the timings match the methodology of the rest of the thesis.

Run
---
    uv run python _findings/gmres_gpu_vs_spsolve_cpu/benchmark.py

Environment variables
---------------------
START_REFINED   first refinement level (default 5)
END_REFINED     last refinement level  (default 7)
PECLET          Peclet number of the advection term (default 30)
TOLERANCE       GMRES relative tolerance (default 1e-5)
"""

import os
import time
from pathlib import Path

import numpy as np

import cupy as cp

from skfem import MeshTri, Basis, ElementTriP1, BilinearForm
from skfem import asm, condense
from skfem.helpers import grad, dot

import scipy.sparse.linalg as spla
from cupyx.scipy.sparse.linalg import gmres
from cupyx.scipy.sparse import csr_matrix as cpx_csr
from skfem.sparse_solvers import (
    get_ilu_preconditioner,
    get_jacobi_preconditioner,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MESH_PATH = REPO_ROOT / "docs" / "examples" / "meshes" / "cylinder_stokes.msh"

PECLET = float(os.getenv("PECLET", "30"))
TOLERANCE = float(os.getenv("TOLERANCE", "1e-5"))
START_REFINED = int(os.getenv("START_REFINED", "5"))
END_REFINED = int(os.getenv("END_REFINED", "7"))
# Preconditioner: "ilu", "jacobi" or "none".  ILU is intentionally NOT the
# default: cupyx's sparse triangular solves are sequential and dominate the
# runtime, killing the speed-up.  Jacobi is essentially free.
PRECOND = os.getenv("PRECOND", "jacobi").lower()
RESTART = int(os.getenv("RESTART", "100"))
MAXITER = int(os.getenv("MAXITER", "4000"))


@BilinearForm
def advection(k, l, m):
    """Advection bilinear form (Stokes flow around a sphere)."""
    r, z = m.x
    u = 1.0
    a = 1.0
    w = r ** 2 + z ** 2
    v_r = ((3 * a * r * z * u) / (4 * w ** 0.5)) * ((a / w) ** 2 - (1 / w))
    v_z = u + ((3 * a * u) / (4 * w ** 0.5)) * (
        (2 * a ** 2 + 3 * r ** 2) / (3 * w) - ((a * r) / w) ** 2 - 2
    )
    return (l * v_r * grad(k)[0] + l * v_z * grad(k)[1]) * 2 * np.pi * r


@BilinearForm
def claplace(u, v, w):
    """Laplace operator in cylindrical coordinates."""
    r = abs(w.x[1])
    return dot(grad(u), grad(v)) * 2 * np.pi * r


def assemble(refine: int):
    """Assemble the condensed advection-diffusion system at a refinement."""
    mesh = MeshTri.load(MESH_PATH).refined(refine)
    basis = Basis(mesh, ElementTriP1())

    interior = basis.complement_dofs(basis.get_dofs({"bottom", "ball"}))

    A = asm(claplace, basis) + PECLET * asm(advection, basis)

    u = basis.zeros()
    u[basis.get_dofs("bottom")] = 1.0
    u[basis.get_dofs("ball")] = 0.0

    Acond, bcond, _, _ = condense(A, x=u, I=interior)
    return Acond.tocsr(), bcond


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    cp.cuda.Stream.null.synchronize()
    return result, time.perf_counter() - start


def make_preconditioner(A_gpu):
    if PRECOND == "ilu":
        return get_ilu_preconditioner(A_gpu)
    if PRECOND == "jacobi":
        return get_jacobi_preconditioner(A_gpu)
    return None


def run_level(refine: int):
    A, b = assemble(refine)
    ndofs = A.shape[0]
    nnz = A.nnz
    bnorm = float(np.linalg.norm(b))

    # --- CPU direct reference ---------------------------------------------
    x_cpu, t_spsolve_cpu = timed(spla.spsolve, A, b)

    # --- GPU GMRES --------------------------------------------------------
    def _to_gpu():
        return (cpx_csr(A, dtype=cp.float64),
                cp.asarray(b, dtype=cp.float64))

    (A_gpu, b_gpu), t_transfer = timed(_to_gpu)

    M, t_ilu = timed(make_preconditioner, A_gpu)

    def _gmres():
        return gmres(A_gpu, b_gpu, rtol=TOLERANCE, restart=RESTART,
                     maxiter=MAXITER, M=M)

    (gmres_out), t_gmres = timed(_gmres)
    x_gpu, info = gmres_out
    x_gmres, t_readback = timed(lambda: cp.asnumpy(x_gpu))

    t_gpu_total = t_transfer + t_ilu + t_gmres + t_readback
    t_gpu_solve_only = t_ilu + t_gmres  # excludes host<->device transfers

    # true relative residual in the 2-norm (the max-rel-error metric is
    # dominated by near-zero solution entries and is misleading here).
    relres = float(np.linalg.norm(A @ x_gmres - b) / (bnorm + 1e-30))

    # cleanup GPU memory
    del A_gpu, b_gpu, M, x_gpu
    cp.get_default_memory_pool().free_all_blocks()

    speedup_total = t_spsolve_cpu / t_gpu_total
    speedup_solve = t_spsolve_cpu / t_gpu_solve_only
    win = "  <== GMRES GPU WINS" if t_gpu_total < t_spsolve_cpu else ""

    print(
        f"r={refine:2d} ndofs={ndofs:>9d} nnz={nnz:>10d} | "
        f"spsolve_cpu={t_spsolve_cpu:8.3f}s | "
        f"[transfer={t_transfer:6.3f} precond={t_ilu:7.3f} "
        f"gmres={t_gmres:7.3f} read={t_readback:5.3f}] | "
        f"gpu_total={t_gpu_total:8.3f}s (x{speedup_total:5.2f}) | "
        f"gpu_solve={t_gpu_solve_only:8.3f}s (x{speedup_solve:5.2f}) | "
        f"info={info:>5d} relres={relres:.1e}{win}",
        flush=True,
    )

    return {
        "refined": refine,
        "ndofs": ndofs,
        "nnz": nnz,
        "spsolve_cpu_s": t_spsolve_cpu,
        "transfer_s": t_transfer,
        "ilu_s": t_ilu,
        "gmres_s": t_gmres,
        "readback_s": t_readback,
        "gpu_total_s": t_gpu_total,
        "gpu_solve_only_s": t_gpu_solve_only,
        "speedup_total": speedup_total,
        "speedup_solve_only": speedup_solve,
        "gmres_info": info,
        "relres": relres,
    }


def main():
    print(
        f"Peclet={PECLET} tol={TOLERANCE} precond={PRECOND} "
        f"restart={RESTART} maxiter={MAXITER} "
        f"levels={START_REFINED}..{END_REFINED}",
        flush=True,
    )

    fields = [
        "refined", "ndofs", "nnz",
        "spsolve_cpu_s", "transfer_s", "ilu_s", "gmres_s", "readback_s",
        "gpu_total_s", "gpu_solve_only_s",
        "speedup_total", "speedup_solve_only", "gmres_info", "relres",
    ]
    rows = []
    for refine in range(START_REFINED, END_REFINED + 1):
        try:
            rows.append(run_level(refine))
        except cp.cuda.memory.OutOfMemoryError:
            print(f"r={refine}: GPU out of memory, stopping.", flush=True)
            cp.get_default_memory_pool().free_all_blocks()
            break
        except MemoryError:
            print(f"r={refine}: host out of memory, stopping.", flush=True)
            break

    suffix = f"_pe{int(PECLET)}_{PRECOND}_r{RESTART}"
    out_csv = Path(__file__).resolve().parent / f"results{suffix}.csv"
    with out_csv.open("w") as fh:
        fh.write(",".join(fields) + "\n")
        for row in rows:
            fh.write(",".join(f"{row[k]}" for k in fields) + "\n")
    print(f"\nSaved {out_csv}", flush=True)


if __name__ == "__main__":
    main()
