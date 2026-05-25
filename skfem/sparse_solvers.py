import functools
import time
from typing import Optional
import cupy as cp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from cupyx.scipy.sparse.linalg import spilu
from cupyx.scipy.sparse.linalg import LinearOperator
from cupyx.scipy.sparse import csr_matrix as cpx_csr
from cupyx.scipy.sparse.linalg import cg, gmres, spsolve as cpx_spsolve
import cupyx.scipy.sparse.linalg as cpx_spla
import os

HALF_PRECISION: bool = bool(int(os.getenv("HALF_PRECISION", "0")))

def _as_dtype(v: cp.ndarray, dtype: cp.dtype) -> cp.ndarray:
    """Return ``v`` as a CuPy array with a target dtype (no copy if possible)."""
    return cp.asarray(v, dtype=dtype)

def timing_decorator(func):
    """Decorator that measures and prints the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"The method {func.__name__} took {elapsed_time:.5f} seconds to execute.")
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Preconditioners
# ---------------------------------------------------------------------------

def get_jacobi_preconditioner(A_gpu: cpx_csr) -> LinearOperator:
    """Returns a Jacobi (diagonal) preconditioner as a LinearOperator."""
    diag = A_gpu.diagonal()
    diag_inv = cp.where(diag != 0, 1.0 / diag, 1.0)

    def matvec(v):
        return diag_inv * v

    return LinearOperator(A_gpu.shape, matvec=matvec)


def get_jacobi_preconditioner_cpu(A_cpu: sp.spmatrix) -> spla.LinearOperator:
    """Returns a Jacobi (diagonal) preconditioner as a LinearOperator for CPU."""
    diag = A_cpu.diagonal()
    diag_inv = np.where(diag != 0, 1.0 / diag, 1.0)

    def matvec(v):
        return diag_inv * v

    return spla.LinearOperator(A_cpu.shape, matvec=matvec)


def get_ilu_preconditioner(A_gpu: cpx_csr) -> LinearOperator:
    """Returns an ILU preconditioner as a LinearOperator.

    Falls back to Jacobi if ILU factorization fails.
    """
    try:
        ilu = spilu(A_gpu)
        return LinearOperator(A_gpu.shape, matvec=ilu.solve)
    except RuntimeError as e:
        print(f"Warning: ILU factorization failed ({e}), falling back to Jacobi preconditioner.")
        return get_jacobi_preconditioner(A_gpu)


# ---------------------------------------------------------------------------
# GPU iterative solvers
# ---------------------------------------------------------------------------

@timing_decorator
def solve_cg_gpu(
    A_gpu: cpx_csr,
    b_gpu: cp.ndarray,
    tol: float = 1e-5,
    M: Optional[LinearOperator] = None,
) -> cp.ndarray:
    """Solves A x = b using CG on the GPU.

    Parameters
    ----------
    M:
        Preconditioner (e.g. from ``get_jacobi_preconditioner``).
        ``None`` means no preconditioning.
    """
    x, info = cg(A_gpu, b_gpu, rtol=tol, M=M)
    if info != 0:
        print(f"Warning: CG did not converge (info={info})")
    return x


@timing_decorator
def solve_gmres_gpu(
    A_gpu: cpx_csr,
    b_gpu: cp.ndarray,
    tol: float = 1e-5,
    M: Optional[LinearOperator] = None,
) -> cp.ndarray:
    """Solves A x = b using GMRES on the GPU.

    Parameters
    ----------
    M:
        Preconditioner (e.g. from ``get_jacobi_preconditioner``).
        ``None`` means no preconditioning.
    """
    x, info = gmres(A_gpu, b_gpu, rtol=tol, M=M)
    if info != 0:
        print(f"Warning: GMRES did not converge (info={info})")
    return x


# ---------------------------------------------------------------------------
# GPU direct solvers
# ---------------------------------------------------------------------------

@timing_decorator
def solve_spsolve_gpu(A_gpu: cpx_csr, b_gpu: cp.ndarray, use_qr: bool = True) -> cp.ndarray:
    """Solves A x = b using cupyx direct sparse solver (GPU)."""
    return cpx_spsolve(A_gpu, b_gpu, use_qr)


@timing_decorator
def solve_superlu_gpu(A_gpu: cpx_csr, b_gpu: cp.ndarray) -> cp.ndarray:
    """Solves A x = b using a sparse LU factorisation on the GPU (cupyx factorized)."""
    solve = cpx_spla.factorized(A_gpu)
    return solve(b_gpu)


# ---------------------------------------------------------------------------
# CPU solvers
# ---------------------------------------------------------------------------

@timing_decorator
def solve_spsolve_cpu(
    A_cpu: sp.spmatrix,
    b_cpu: np.ndarray,
) -> np.ndarray:
    """Solves A x = b using SciPy's direct sparse solver (CPU)."""
    return spla.spsolve(A_cpu, b_cpu)


@timing_decorator
def solve_cg_cpu(
    A_cpu: sp.spmatrix,
    b_cpu: np.ndarray,
    tol: float = 1e-5,
    M: Optional[spla.LinearOperator] = None,
) -> np.ndarray:
    """Solves A x = b using SciPy's CG solver (CPU).

    Parameters
    ----------
    A_cpu:
        System matrix as a SciPy sparse matrix.
    b_cpu:
        Right-hand side vector as a NumPy array.
    tol:
        Solver tolerance.
    M:
        Preconditioner (e.g. from ``get_jacobi_preconditioner_cpu``).
        ``None`` means no preconditioning.
    """
    x, info = spla.cg(A_cpu, b_cpu, rtol=tol, M=M)
    if info != 0:
        print(f"Warning: CG CPU did not converge (info={info})")
    return x


@timing_decorator
def load_to_gpu(A, b):
    dtype = cp.float32 if HALF_PRECISION else cp.float64
    return cpx_csr(A, dtype=dtype), cp.asarray(b, dtype=dtype)

@timing_decorator
def read_from_gpu(x_gpu):
    return cp.asnumpy(x_gpu)
