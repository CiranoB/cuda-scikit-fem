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


def get_block_jacobi_preconditioner(
    A_gpu: cpx_csr,
    block_size: int = 8,
) -> LinearOperator:
    """Returns a block-Jacobi preconditioner as a LinearOperator.

    The matrix is split into contiguous diagonal blocks of ``block_size`` and
    every block is inverted once with a single batched dense inversion.  Each
    application is therefore one parallel batched mat-vec, avoiding the
    sequential triangular solves that make ILU-type preconditioners slow on the
    GPU.  When ``block_size == 1`` this reduces to the Jacobi preconditioner.
    """
    n = A_gpu.shape[0]
    bs = max(1, int(block_size))
    n_blocks = (n + bs - 1) // bs
    padded = n_blocks * bs

    coo = A_gpu.tocoo()
    coo.sum_duplicates()
    same_block = (coo.row // bs) == (coo.col // bs)
    blk = coo.row[same_block] // bs
    local_row = coo.row[same_block] % bs
    local_col = coo.col[same_block] % bs

    blocks = cp.zeros((n_blocks, bs, bs), dtype=A_gpu.dtype)
    blocks[blk, local_row, local_col] = coo.data[same_block]

    # Padded rows of the trailing block act as identity so the batched inverse
    # is well defined when ``n`` is not a multiple of ``block_size``.
    if padded != n:
        pad_local = cp.arange(n % bs, bs)
        blocks[n_blocks - 1, pad_local, pad_local] = 1.0

    # Keep degenerate (all-zero) diagonal entries invertible; a no-op for SPD
    # matrices whose diagonal is strictly positive.
    diag_idx = cp.arange(bs)
    block_diag = blocks[:, diag_idx, diag_idx]
    blocks[:, diag_idx, diag_idx] = cp.where(block_diag == 0, 1.0, block_diag)

    blocks_inv = cp.linalg.inv(blocks)

    def matvec(v):
        vv = v if padded == n else cp.concatenate(
            [v, cp.zeros(padded - n, dtype=v.dtype)])
        yb = cp.einsum("kij,kj->ki", blocks_inv, vv.reshape(n_blocks, bs))
        return yb.reshape(padded)[:n]

    return LinearOperator(A_gpu.shape, matvec=matvec)


def get_polynomial_preconditioner(
    A_gpu: cpx_csr,
    degree: int = 3,
) -> LinearOperator:
    """Returns a symmetric polynomial (Neumann series) preconditioner.

    Approximates ``A^{-1}`` by a truncated Neumann series of the symmetrically
    scaled operator ``B = D^{-1/2} A D^{-1/2}``, so that the preconditioner
    stays symmetric (as required by CG).  Each application is a short sequence
    of sparse mat-vecs and diagonal scalings with no triangular solves, which
    maps well to the GPU.  ``degree == 0`` reduces to symmetric Jacobi.
    """
    diag = A_gpu.diagonal()
    d_inv_sqrt = cp.where(diag > 0, 1.0 / cp.sqrt(diag), 1.0)
    m = max(0, int(degree))

    def matvec(v):
        z = d_inv_sqrt * v
        s = z
        for _ in range(m):
            # (I - B) s = s - D^{-1/2} A D^{-1/2} s
            s = z + (s - d_inv_sqrt * (A_gpu @ (d_inv_sqrt * s)))
        return d_inv_sqrt * s

    return LinearOperator(A_gpu.shape, matvec=matvec)


def get_block_jacobi_preconditioner_cpu(
    A_cpu: sp.spmatrix,
    block_size: int = 8,
) -> spla.LinearOperator:
    """CPU counterpart of :func:`get_block_jacobi_preconditioner`."""
    n = A_cpu.shape[0]
    bs = max(1, int(block_size))
    n_blocks = (n + bs - 1) // bs
    padded = n_blocks * bs

    coo = A_cpu.tocoo()
    coo.sum_duplicates()
    same_block = (coo.row // bs) == (coo.col // bs)
    blk = coo.row[same_block] // bs
    local_row = coo.row[same_block] % bs
    local_col = coo.col[same_block] % bs

    blocks = np.zeros((n_blocks, bs, bs), dtype=A_cpu.dtype)
    blocks[blk, local_row, local_col] = coo.data[same_block]

    if padded != n:
        pad_local = np.arange(n % bs, bs)
        blocks[n_blocks - 1, pad_local, pad_local] = 1.0

    diag_idx = np.arange(bs)
    block_diag = blocks[:, diag_idx, diag_idx]
    blocks[:, diag_idx, diag_idx] = np.where(block_diag == 0, 1.0, block_diag)

    blocks_inv = np.linalg.inv(blocks)

    def matvec(v):
        vv = v if padded == n else np.concatenate(
            [v, np.zeros(padded - n, dtype=v.dtype)])
        yb = np.einsum("kij,kj->ki", blocks_inv, vv.reshape(n_blocks, bs))
        return yb.reshape(padded)[:n]

    return spla.LinearOperator(A_cpu.shape, matvec=matvec)


def get_polynomial_preconditioner_cpu(
    A_cpu: sp.spmatrix,
    degree: int = 3,
) -> spla.LinearOperator:
    """CPU counterpart of :func:`get_polynomial_preconditioner`."""
    diag = A_cpu.diagonal()
    d_inv_sqrt = np.where(diag > 0, 1.0 / np.sqrt(diag), 1.0)
    m = max(0, int(degree))

    def matvec(v):
        z = d_inv_sqrt * v
        s = z
        for _ in range(m):
            # (I - B) s = s - D^{-1/2} A D^{-1/2} s
            s = z + (s - d_inv_sqrt * (A_cpu @ (d_inv_sqrt * s)))
        return d_inv_sqrt * s

    return spla.LinearOperator(A_cpu.shape, matvec=matvec)


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
def solve_spsolve_gpu(A_gpu: cpx_csr, b_gpu: cp.ndarray) -> cp.ndarray:
    """Solves A x = b using cupyx direct sparse solver (GPU)."""
    return cpx_spsolve(A_gpu, b_gpu)


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
