"""Example 33: does the GPU CG arithmetic precision matter?

The rest of the thesis runs every solver in IEEE-754 double precision
(``float64``).  This experiment isolates a single question: for the Example 33
SPD system, what happens to the **GPU** Conjugate Gradient path if the matrix,
right-hand side, preconditioner, and iterates are stored and computed in
**single precision** (``float32``) or **half precision** (``float16``) instead?

For a growing tensor tetrahedral mesh the lowest-order Nedelec system is
assembled and the tangential boundary condition condensed, giving a symmetric
positive definite system ``A x = b``.  The double-precision CPU direct solve
``scipy.sparse.linalg.spsolve`` provides the trusted reference ``x_ref``.  For
each mesh size and each precision the harness runs GPU CG (unpreconditioned and
Jacobi-preconditioned) and records, per case:

* the host->device transfer time (less data to move at lower precision),
* the preconditioner build time and the CG solve time,
* the device->host readback time,
* the CG iteration count and convergence status,
* the attained 2-norm relative residual ``||A x - b|| / ||b||`` **recomputed in
  float64** against the original double-precision operator, so the number is a
  faithful measure of the true error rather than the (optimistic) residual the
  low-precision solver believes it reached, and
* the mean / max absolute difference from the double-precision ``spsolve``
  reference.

A second, smaller probe (``tol_floor.csv``) fixes one representative mesh and
sweeps the requested CG tolerance to expose the accuracy floor of each
precision: single precision cannot drive the residual much below its unit
round-off (~1e-7), whereas double precision keeps converging.

Run
---
    PYTHONPATH=. uv run python _findings/ex33_precision/benchmark.py

Useful environment variables
----------------------------
PRECISION_PTS      comma-separated points/direction  (default 10,12,14,16,19,23)
FLOOR_PTS          mesh for the tolerance-floor probe (default 16)
CG_MAXITER         CG iteration cap                   (default 5000)
TOLERANCE          CG relative tolerance for the sweep(default 1e-5)
CASE_TIMEOUT_S     per-solve wall budget              (default 120)
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

import cupy as cp
from cupyx.scipy.sparse import csr_matrix as cpx_csr
from cupyx.scipy.sparse.linalg import cg as cg_gpu, LinearOperator

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


OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = OUTPUT_DIR / "results.csv"
FLOOR_CSV = OUTPUT_DIR / "tol_floor.csv"

CASES = ("none", "jacobi")

# (label, numpy dtype) for each precision under test.  float16 is attempted but
# is expected to be unsupported by the cuSPARSE-backed sparse kernels; the
# harness records the failure honestly rather than skipping it.
PRECISIONS = (
    ("fp64", np.float64),
    ("fp32", np.float32),
    ("fp16", np.float16),
)

PRECISION_PTS = [int(p) for p in
                 os.getenv("PRECISION_PTS", "10,12,14,16,19,23").split(",")]
FLOOR_PTS = int(os.getenv("FLOOR_PTS", "16"))
CG_MAXITER = int(os.getenv("CG_MAXITER", "5000"))
TOLERANCE = float(os.getenv("TOLERANCE", "1e-5"))
CASE_TIMEOUT_S = float(os.getenv("CASE_TIMEOUT_S", "120"))


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
    return captured["A"], captured["b"]


def _jacobi_precond(A_gpu):
    """Jacobi preconditioner built in the dtype of ``A_gpu``."""
    diag = A_gpu.diagonal()
    diag_inv = cp.where(diag != 0, (1.0 / diag).astype(A_gpu.dtype), A_gpu.dtype.type(1.0))

    def matvec(v):
        return diag_inv * v

    return LinearOperator(A_gpu.shape, matvec=matvec, dtype=A_gpu.dtype)


def _callback_factory(state: dict):
    def _cb(_xk):
        state["iters"] += 1
        if time.perf_counter() - state["start"] > CASE_TIMEOUT_S:
            raise _Budget()
    return _cb


def _to_gpu(A_cpu, b_cpu, dtype):
    """Move ``A``/``b`` to the GPU in ``dtype``; time the transfer."""
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    A_gpu = cpx_csr(A_cpu.astype(dtype))
    b_gpu = cp.asarray(b_cpu, dtype=dtype)
    cp.cuda.Stream.null.synchronize()
    return A_gpu, b_gpu, time.perf_counter() - t0


def run_gpu_cg(A_gpu, b_gpu, case, tol=TOLERANCE):
    """Run GPU CG for one (precision, case); return raw timing/iteration data."""
    t0 = time.perf_counter()
    M = _jacobi_precond(A_gpu) if case == "jacobi" else None
    cp.cuda.Stream.null.synchronize()
    precond_s = time.perf_counter() - t0

    state = {"iters": 0, "start": time.perf_counter()}
    status = "converged"
    t0 = state["start"]
    try:
        x_gpu, info = cg_gpu(
            A_gpu, b_gpu, rtol=tol, maxiter=CG_MAXITER, M=M,
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
    x = None if x_gpu is None else cp.asnumpy(x_gpu).astype(np.float64)
    readback_s = time.perf_counter() - t0

    del M, x_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return {
        "precond_s": precond_s,
        "solve_s": solve_s,
        "readback_s": readback_s,
        "status": status,
        "iters": state["iters"],
        "x": x,
    }


def _accuracy(A64, b64, bnorm, x_ref, x):
    """True (float64) residual and difference from the direct reference."""
    if x is None or not np.all(np.isfinite(x)):
        return float("nan"), float("nan"), float("nan")
    relres = float(np.linalg.norm(A64 @ x - b64) / bnorm)
    diff = np.abs(x - x_ref)
    return relres, float(diff.mean()), float(diff.max())


def _fieldnames():
    names = ["pts_per_dir", "ndofs", "nnz", "tol", "maxiter"]
    for label, _ in PRECISIONS:
        for case in CASES:
            prefix = f"{label}_{case}_"
            names += [prefix + s for s in (
                "transfer_s", "precond_s", "solve_s", "readback_s", "status",
                "converged", "iters", "relres", "mean_abs_diff", "max_abs_diff")]
    return names


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Warm up the CUDA context and the fp64/fp32 CG code paths on a tiny system
    # so one-time JIT / allocator costs are not charged to the first measurement.
    import scipy.sparse as _sp
    _Aw = (_sp.eye(64) * 2.0).tocsr()
    _bw = np.ones(64)
    for _, dt in PRECISIONS[:2]:
        _Ag, _bg, _ = _to_gpu(_Aw, _bw, dt)
        cg_gpu(_Ag, _bg, rtol=1e-5, maxiter=50)
        del _Ag, _bg
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    rows: list[dict] = []
    print(f"[plan] precision sweep pts={PRECISION_PTS}", flush=True)
    for pts in PRECISION_PTS:
        A, b = assemble_condensed(pts)
        A64 = A.astype(np.float64)
        b64 = b.astype(np.float64)
        ndofs = A.shape[0]
        nnz = int(A.nnz)
        bnorm = float(np.linalg.norm(b64)) or 1.0

        t0 = time.perf_counter()
        x_ref = spla.spsolve(A64, b64)
        spsolve_s = time.perf_counter() - t0
        print(f"[pts={pts}] ndofs={ndofs} nnz={nnz} spsolve={spsolve_s:.3f}s",
              flush=True)

        row = {"pts_per_dir": pts, "ndofs": ndofs, "nnz": nnz,
               "tol": TOLERANCE, "maxiter": CG_MAXITER}

        for label, dtype in PRECISIONS:
            try:
                A_gpu, b_gpu, transfer_s = _to_gpu(A, b, dtype)
            except Exception as exc:  # noqa: BLE001 - report, don't crash
                short = f"{type(exc).__name__}: {exc}".splitlines()[0][:60]
                print(f"    {label:4s} transfer unsupported ({short})", flush=True)
                for case in CASES:
                    p = f"{label}_{case}_"
                    row.update({p + "transfer_s": float("nan"),
                                p + "precond_s": float("nan"),
                                p + "solve_s": float("nan"),
                                p + "readback_s": float("nan"),
                                p + "status": "unsupported", p + "converged": 0,
                                p + "iters": 0, p + "relres": float("nan"),
                                p + "mean_abs_diff": float("nan"),
                                p + "max_abs_diff": float("nan")})
                continue

            for case in CASES:
                try:
                    r = run_gpu_cg(A_gpu, b_gpu, case)
                    relres, mean_abs, max_abs = _accuracy(
                        A64, b64, bnorm, x_ref, r["x"])
                    status = r["status"]
                    iters = r["iters"]
                    precond_s, solve_s, readback_s = (
                        r["precond_s"], r["solve_s"], r["readback_s"])
                except Exception as exc:  # noqa: BLE001
                    short = f"{type(exc).__name__}: {exc}".splitlines()[0][:60]
                    print(f"    {label:4s} {case:8s} unsupported ({short})",
                          flush=True)
                    status = "unsupported"
                    iters = 0
                    precond_s = solve_s = readback_s = float("nan")
                    relres = mean_abs = max_abs = float("nan")

                converged = int(status == "converged"
                                and np.isfinite(relres) and relres <= TOLERANCE)
                p = f"{label}_{case}_"
                row.update({
                    p + "transfer_s": transfer_s, p + "precond_s": precond_s,
                    p + "solve_s": solve_s, p + "readback_s": readback_s,
                    p + "status": status, p + "converged": converged,
                    p + "iters": iters, p + "relres": relres,
                    p + "mean_abs_diff": mean_abs, p + "max_abs_diff": max_abs,
                })
                print(f"    {label:4s} {case:8s} {status:11s} "
                      f"solve={solve_s if np.isfinite(solve_s) else float('nan'):.4f}s "
                      f"iters={iters} relres={relres:.2e} "
                      f"maxdiff={max_abs:.2e}", flush=True)

            del A_gpu, b_gpu
            cp.get_default_memory_pool().free_all_blocks()

        rows.append(row)
        with RESULTS_CSV.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_fieldnames())
            writer.writeheader()
            writer.writerows(rows)

    print(f"[done] {len(rows)} sizes written to {RESULTS_CSV}", flush=True)

    _tolerance_floor_probe()


def _tolerance_floor_probe() -> None:
    """Fix one mesh; sweep the CG tolerance for fp64 and fp32 (Jacobi)."""
    A, b = assemble_condensed(FLOOR_PTS)
    A64 = A.astype(np.float64)
    b64 = b.astype(np.float64)
    bnorm = float(np.linalg.norm(b64)) or 1.0
    x_ref = spla.spsolve(A64, b64)
    ndofs = A.shape[0]
    print(f"[floor] pts={FLOOR_PTS} ndofs={ndofs} tolerance sweep (Jacobi)",
          flush=True)

    tols = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    out_rows = []
    for label, dtype in PRECISIONS[:2]:  # fp64, fp32
        A_gpu, b_gpu, _ = _to_gpu(A, b, dtype)
        for tol in tols:
            r = run_gpu_cg(A_gpu, b_gpu, "jacobi", tol=tol)
            relres, mean_abs, max_abs = _accuracy(A64, b64, bnorm, x_ref, r["x"])
            out_rows.append({
                "precision": label, "ndofs": ndofs, "requested_tol": tol,
                "status": r["status"], "iters": r["iters"],
                "attained_relres": relres, "mean_abs_diff": mean_abs,
                "max_abs_diff": max_abs, "solve_s": r["solve_s"],
            })
            print(f"    {label:4s} tol={tol:.0e} {r['status']:9s} "
                  f"iters={r['iters']} relres={relres:.2e}", flush=True)
        del A_gpu, b_gpu
        cp.get_default_memory_pool().free_all_blocks()

    with FLOOR_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[done] tolerance floor written to {FLOOR_CSV}", flush=True)


if __name__ == "__main__":
    main()
