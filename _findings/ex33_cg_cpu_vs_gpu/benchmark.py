"""Example 33: CPU CG versus GPU CG with four preconditioners.

This benchmark reproduces (and extends) the Example 33 solver study used in the
thesis.  For a growing tensor tetrahedral mesh it assembles the lowest-order
Nedelec system, condenses the tangential boundary condition, and then times the
solve of the resulting symmetric positive definite system

    A x = b

with

* the CPU direct solver ``scipy.sparse.linalg.spsolve`` (reference), and
* the Conjugate Gradient method on **both** the CPU and the GPU, each with the
  same four preconditioner variants:

      none, jacobi, block_jacobi, polynomial

(The ``ilu0`` variant used by the whole-suite campaign is deliberately excluded
here: its sequential triangular solves have no CPU/GPU-symmetric counterpart and
it is not part of this device-vs-device comparison.)

For every (mesh level, device, case) it records the preconditioner build time,
the CG solve time, the device->host readback time (GPU only), whether CG
converged, the iteration count, the 2-norm relative residual
``||A x - b|| / ||b||`` and the absolute difference from the CPU ``spsolve``
reference.  The per-level host->device transfer time is recorded once and shared
by the GPU cases so that an end-to-end GPU cost can be reconstructed as
``transfer + precond + solve + readback``.

Each mesh level runs in the same process; the sweep grows the mesh until the CPU
``spsolve`` reference exceeds ``SPSOLVE_CAP_S`` seconds, a global wall-clock
budget is exhausted, or a level fails (e.g. GPU/host out of memory).  Results are
written incrementally to ``results.csv`` so a partial sweep is never lost.

Run
---
    PYTHONPATH=. uv run python _findings/ex33_cg_cpu_vs_gpu/benchmark.py

Useful environment variables
----------------------------
START_LEVEL        first mesh level (pts/dir = 14*level+1)   (default 1)
MAX_LEVEL          global ceiling on the mesh level          (default 40)
SPSOLVE_CAP_S      stop after a level whose spsolve exceeds  (default 120)
GLOBAL_BUDGET_S    wall-clock cap for the whole sweep        (default 3600)
CG_CASE_TIMEOUT_S  per-solve wall budget (CPU and GPU)       (default 120)
CG_MAXITER         CG iteration cap                          (default 5000)
CG_BLOCK_SIZE      block size for block_jacobi               (default 8)
CG_POLY_DEGREE     degree of the polynomial preconditioner   (default 3)
TOLERANCE          CG relative tolerance                     (default 1e-5)
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

import cupy as cp
from cupyx.scipy.sparse.linalg import cg as cg_gpu

from skfem import (
    MeshTet,
    Basis,
    ElementTetN0,
    BilinearForm,
    LinearForm,
    asm,
    condense,
    solve,
)
from skfem.helpers import curl, dot
from skfem.sparse_solvers import (
    load_to_gpu,
    get_jacobi_preconditioner,
    get_jacobi_preconditioner_cpu,
    get_block_jacobi_preconditioner,
    get_block_jacobi_preconditioner_cpu,
    get_polynomial_preconditioner,
    get_polynomial_preconditioner_cpu,
)


OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = OUTPUT_DIR / "results.csv"

CASES = ("none", "jacobi", "block_jacobi", "polynomial")

# The mesh is grown as a geometric sequence of points-per-direction so that the
# number of degrees of freedom (~ pts**3 in 3D) increases by a modest factor at
# every step, giving enough intermediate sizes for a clean scaling plot.
START_PTS = int(os.getenv("START_PTS", "6"))
MAX_PTS = int(os.getenv("MAX_PTS", "120"))
PTS_GROWTH = float(os.getenv("PTS_GROWTH", "1.18"))
MAX_LEVELS = int(os.getenv("MAX_LEVELS", "40"))
SPSOLVE_CAP_S = float(os.getenv("SPSOLVE_CAP_S", "120"))
GLOBAL_BUDGET_S = float(os.getenv("GLOBAL_BUDGET_S", "3600"))
CG_CASE_TIMEOUT_S = float(os.getenv("CG_CASE_TIMEOUT_S", "120"))
CG_MAXITER = int(os.getenv("CG_MAXITER", "5000"))
CG_BLOCK_SIZE = int(os.getenv("CG_BLOCK_SIZE", "8"))
CG_POLY_DEGREE = int(os.getenv("CG_POLY_DEGREE", "3"))
TOLERANCE = float(os.getenv("TOLERANCE", "1e-5"))


class _Budget(Exception):
    """Raised from a CG callback when the per-solve wall budget is exceeded."""


def _rhs(x, y, z):
    return np.array([
        x * y * (1 - y**2) * (1 - z**2) + 2 * x * y * (1 - z**2),
        y**2 * (1 - x**2) * (1 - z**2) + (1 - y**2) * (2 - x**2 - z**2),
        y * z * (1 - x**2) * (1 - y**2) + 2 * y * z * (1 - x**2),
    ])


@BilinearForm
def _dudv(E, v, w):
    return dot(curl(E), curl(v)) + dot(E, v)


@LinearForm
def _fv(v, w):
    return dot(_rhs(*w.x), v)


def pts_sequence() -> list[int]:
    """Strictly increasing points-per-direction with geometric growth."""
    seq: list[int] = []
    p = float(START_PTS)
    while int(round(p)) <= MAX_PTS and len(seq) < MAX_LEVELS:
        ip = int(round(p))
        if not seq or ip > seq[-1]:
            seq.append(ip)
        p = max(p * PTS_GROWTH, p + 1)
    return seq


def assemble_condensed(pts: int):
    """Assemble Example 33 with ``pts`` points/direction; return condensed (A, b)."""
    m = MeshTet.init_tensor(
        np.linspace(-1, 1, pts),
        np.linspace(-1, 1, pts),
        np.linspace(-1, 1, pts),
    )
    basis = Basis(m, ElementTetN0())
    A = asm(_dudv, basis)
    b = asm(_fv, basis)
    D = basis.get_dofs()

    captured: dict = {}

    def _capture(Ac, bc, **_kw):
        captured["A"] = Ac.tocsr()
        captured["b"] = np.asarray(bc)
        return np.zeros_like(captured["b"])

    solve(*condense(A, b, D=D), solver=_capture)
    return captured["A"], captured["b"], pts


def _make_cpu_precond(case: str, A):
    if case == "none":
        return None
    if case == "jacobi":
        return get_jacobi_preconditioner_cpu(A)
    if case == "block_jacobi":
        return get_block_jacobi_preconditioner_cpu(A, block_size=CG_BLOCK_SIZE)
    if case == "polynomial":
        return get_polynomial_preconditioner_cpu(A, degree=CG_POLY_DEGREE)
    raise ValueError(case)


def _make_gpu_precond(case: str, A_gpu):
    if case == "none":
        return None
    if case == "jacobi":
        return get_jacobi_preconditioner(A_gpu)
    if case == "block_jacobi":
        return get_block_jacobi_preconditioner(A_gpu, block_size=CG_BLOCK_SIZE)
    if case == "polynomial":
        return get_polynomial_preconditioner(A_gpu, degree=CG_POLY_DEGREE)
    raise ValueError(case)


def _callback_factory(state: dict):
    def _cb(_xk):
        state["iters"] += 1
        if time.perf_counter() - state["start"] > CG_CASE_TIMEOUT_S:
            raise _Budget()
    return _cb


def run_cpu_cg(A, b, x_ref, bnorm, case):
    """Time CPU CG for one preconditioner case; return a metrics dict."""
    t0 = time.perf_counter()
    M = _make_cpu_precond(case, A)
    precond_s = time.perf_counter() - t0

    state = {"iters": 0, "start": time.perf_counter()}
    status = "converged"
    t0 = state["start"]
    try:
        x, info = spla.cg(
            A, b, rtol=TOLERANCE, maxiter=CG_MAXITER, M=M,
            callback=_callback_factory(state),
        )
        if info != 0:
            status = "maxiter"
    except _Budget:
        x = None
        status = "timeout"
    solve_s = time.perf_counter() - t0

    return _metrics(A, b, x, x_ref, bnorm, case, precond_s, solve_s, 0.0,
                    status, state["iters"])


def run_gpu_cg(A_gpu, b_gpu, A_cpu, b_cpu, x_ref, bnorm, case):
    """Time GPU CG for one preconditioner case; return a metrics dict."""
    t0 = time.perf_counter()
    M = _make_gpu_precond(case, A_gpu)
    cp.cuda.Stream.null.synchronize()
    precond_s = time.perf_counter() - t0

    state = {"iters": 0, "start": time.perf_counter()}
    status = "converged"
    t0 = state["start"]
    try:
        x_gpu, info = cg_gpu(
            A_gpu, b_gpu, rtol=TOLERANCE, maxiter=CG_MAXITER, M=M,
            callback=_callback_factory(state),
        )
        if info != 0:
            status = "maxiter"
    except _Budget:
        x_gpu = None
        status = "timeout"
    cp.cuda.Stream.null.synchronize()
    solve_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    x = None if x_gpu is None else cp.asnumpy(x_gpu)
    readback_s = time.perf_counter() - t0

    del M, x_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return _metrics(A_cpu, b_cpu, x, x_ref, bnorm, case, precond_s, solve_s,
                    readback_s, status, state["iters"])


def _metrics(A, b, x, x_ref, bnorm, case, precond_s, solve_s, readback_s,
             status, iters):
    if x is None:
        relres = float("nan")
        mean_abs = float("nan")
        max_abs = float("nan")
    else:
        relres = float(np.linalg.norm(A @ x - b) / bnorm)
        diff = np.abs(x - x_ref)
        mean_abs = float(diff.mean())
        max_abs = float(diff.max())
    return {
        "case": case,
        "precond_s": precond_s,
        "solve_s": solve_s,
        "readback_s": readback_s,
        "status": status,
        "converged": int(status == "converged"),
        "iters": iters,
        "relres": relres,
        "mean_abs_diff": mean_abs,
        "max_abs_diff": max_abs,
    }


def _fieldnames():
    names = ["level", "pts_per_dir", "ndofs", "nnz",
             "spsolve_cpu_s", "transfer_s", "tol", "maxiter"]
    for device in ("cpu", "gpu"):
        for case in CASES:
            prefix = f"{device}_{case}_"
            names += [prefix + s for s in (
                "precond_s", "solve_s", "readback_s", "status", "converged",
                "iters", "relres", "mean_abs_diff", "max_abs_diff")]
    return names


def _flatten(base, cpu_rows, gpu_rows):
    row = dict(base)
    for device, rows in (("cpu", cpu_rows), ("gpu", gpu_rows)):
        for m in rows:
            prefix = f"{device}_{m['case']}_"
            for key in ("precond_s", "solve_s", "readback_s", "status",
                        "converged", "iters", "relres", "mean_abs_diff",
                        "max_abs_diff"):
                row[prefix + key] = m[key]
    return row


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Warm up the CUDA context so the first transfer time is genuine.
    _warm = cp.zeros(1) + 1
    cp.cuda.Stream.null.synchronize()
    del _warm

    # Warm up the CPU and GPU CG code paths on a tiny system so that one-time
    # import / JIT / allocator costs are not charged to the first measured
    # level (which would otherwise inflate its ``solve_s``).
    import scipy.sparse as _sp
    _Aw = (_sp.eye(64) * 2.0).tocsr()
    _bw = np.ones(64)
    spla.cg(_Aw, _bw, rtol=1e-6, maxiter=50)
    _Aw_gpu, _bw_gpu = load_to_gpu(_Aw, _bw)
    cg_gpu(_Aw_gpu, _bw_gpu, rtol=1e-6, maxiter=50)
    cp.cuda.Stream.null.synchronize()
    del _Aw_gpu, _bw_gpu
    cp.get_default_memory_pool().free_all_blocks()

    rows: list[dict] = []
    sweep_start = time.perf_counter()

    sequence = pts_sequence()
    print(f"[plan] {len(sequence)} mesh levels: pts={sequence}", flush=True)
    for level, pts_val in enumerate(sequence, start=1):
        if time.perf_counter() - sweep_start > GLOBAL_BUDGET_S:
            print(f"[stop] global budget {GLOBAL_BUDGET_S}s reached")
            break

        try:
            A, b, pts = assemble_condensed(pts_val)
        except MemoryError:
            print(f"[stop] level={level}: host out of memory during assembly")
            break

        ndofs = A.shape[0]
        nnz = int(A.nnz)
        bnorm = float(np.linalg.norm(b)) or 1.0

        t0 = time.perf_counter()
        x_ref = spla.spsolve(A, b)
        spsolve_s = time.perf_counter() - t0

        print(f"[level {level}] pts={pts} ndofs={ndofs} nnz={nnz} "
              f"spsolve={spsolve_s:.4f}s", flush=True)

        cpu_rows = []
        for case in CASES:
            m = run_cpu_cg(A, b, x_ref, bnorm, case)
            cpu_rows.append(m)
            print(f"    cpu  {case:12s} {m['status']:9s} "
                  f"solve={m['solve_s']:.4f}s iters={m['iters']} "
                  f"relres={m['relres']:.2e}", flush=True)

        gpu_rows = []
        try:
            t0 = time.perf_counter()
            A_gpu, b_gpu = load_to_gpu(A, b)
            cp.cuda.Stream.null.synchronize()
            transfer_s = time.perf_counter() - t0
            try:
                for case in CASES:
                    m = run_gpu_cg(A_gpu, b_gpu, A, b, x_ref, bnorm, case)
                    gpu_rows.append(m)
                    print(f"    gpu  {case:12s} {m['status']:9s} "
                          f"solve={m['solve_s']:.4f}s iters={m['iters']} "
                          f"relres={m['relres']:.2e}", flush=True)
            finally:
                del A_gpu, b_gpu
                cp.get_default_memory_pool().free_all_blocks()
        except cp.cuda.memory.OutOfMemoryError:
            print(f"[stop] level={level}: GPU out of memory")
            transfer_s = float("nan")

        base = {
            "level": level, "pts_per_dir": pts, "ndofs": ndofs, "nnz": nnz,
            "spsolve_cpu_s": spsolve_s, "transfer_s": transfer_s,
            "tol": TOLERANCE, "maxiter": CG_MAXITER,
        }
        rows.append(_flatten(base, cpu_rows, gpu_rows))

        with RESULTS_CSV.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_fieldnames())
            writer.writeheader()
            writer.writerows(rows)

        if not gpu_rows:
            print("[stop] GPU failed on this level")
            break
        if spsolve_s > SPSOLVE_CAP_S:
            print(f"[stop] spsolve {spsolve_s:.1f}s exceeded cap "
                  f"{SPSOLVE_CAP_S}s")
            break

    print(f"[done] {len(rows)} levels written to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
