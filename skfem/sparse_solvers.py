import functools
import time
import cupy as cp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from cupyx.scipy.sparse.linalg import spilu  # ILU na GPU
from cupyx.scipy.sparse.linalg import LinearOperator
from cupyx.scipy.sparse import csr_matrix as cpx_csr
from cupyx.scipy.sparse.linalg import cg, spsolve as cpx_spsolve
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

def get_jacobi_preconditioner(A_gpu):
    """Returns a Jacobi (diagonal) preconditioner as a LinearOperator."""
    diag = A_gpu.diagonal()
    # Avoid division by zero
    diag_inv = cp.where(diag != 0, 1.0 / diag, 1.0)
    
    def matvec(v):
        return diag_inv * v

    return LinearOperator(A_gpu.shape, matvec=matvec)

@timing_decorator
def solve_spsolve_gpu(A_gpu: cpx_csr, b_gpu: cp.ndarray) -> cp.ndarray:
    """Solves A x = b using cupyx direct sparse solver (GPU)."""
    return cpx_spsolve(A_gpu, b_gpu)


@timing_decorator
def solve_cg_with_jacobi_preconditioner(
    A_gpu: cpx_csr,
    b_gpu: cp.ndarray,
    tol: float = 1e-5,
) -> cp.ndarray:
    """Solves A x = b using CG with Jacobi preconditioner (GPU).

    Parameters
    ----------
    A_gpu:
        System matrix in CuPy CSR format.
    b_gpu:
        Right-hand side vector on the GPU.
    tol:
        Solver tolerance (e.g. 1, 1e-3, 1e-5).
    """
    M = get_jacobi_preconditioner(A_gpu)
    x, info = cg(A_gpu, b_gpu, rtol=tol, M=M)
    if info != 0:
        print(f"Warning: CG (with Jacobi preconditioner) did not converge (info={info})")
    return x


@timing_decorator
def solve_cg_with_ilu_preconditioner(
    A_gpu: cpx_csr,
    b_gpu: cp.ndarray,
    tol: float = 1e-5,
) -> cp.ndarray:
    """Solves A x = b using CG with ILU preconditioner (GPU).

    Parameters
    ----------
    A_gpu:
        System matrix in CuPy CSR format.
    b_gpu:
        Right-hand side vector on the GPU.
    tol:
        Solver tolerance (e.g. 1, 1e-3, 1e-5).
    """
    try:
        ilu = spilu(A_gpu)
        M = LinearOperator(A_gpu.shape, matvec=ilu.solve)
    except RuntimeError as e:
        print(f"Warning: ILU factorization failed ({e}), falling back to Jacobi preconditioner.")
        M = get_jacobi_preconditioner(A_gpu)

    x, info = cg(A_gpu, b_gpu, rtol=tol, M=M)
    if info != 0:
        print(f"Warning: CG (with ILU preconditioner) did not converge (info={info})")
    return x


@timing_decorator
def solve_cg_without_jacobi_preconditioner(
    A_gpu: cpx_csr,
    b_gpu: cp.ndarray,
    tol: float = 1e-5,
) -> cp.ndarray:
    """Solves A x = b using CG with Jacobi preconditioner (GPU).

    Parameters
    ----------
    A_gpu:
        System matrix in CuPy CSR format.
    b_gpu:
        Right-hand side vector on the GPU.
    tol:
        Solver tolerance (e.g. 1, 1e-3, 1e-5).
    """
    x, info = cg(A_gpu, b_gpu, rtol=tol)
    if info != 0:
        print(f"Warning: CG (without Jacobi preconditioner) did not converge (info={info})")
    return x


# OK
@timing_decorator
def solve_superlu_gpu(A_gpu: cpx_csr, b_gpu: cp.ndarray) -> cp.ndarray:
    """Solves A x = b using a sparse LU factorisation on the GPU (cupyx factorized)."""
    solve = cpx_spla.factorized(A_gpu)
    return solve(b_gpu)


# OK
@timing_decorator
def solve_spsolve_cpu(
    A_cpu: sp.spmatrix,
    b_cpu: np.ndarray,
) -> np.ndarray:
    """Solves A x = b using SciPy's direct sparse solver (CPU).

    Parameters
    ----------
    A_cpu:
        System matrix as a SciPy sparse matrix.
    b_cpu:
        Right-hand side vector as a NumPy array.
    """
    return spla.spsolve(A_cpu, b_cpu)

def is_symmetric(A: sp.spmatrix, tol: float = 1e-10) -> bool:
    """Check whether a SciPy sparse matrix is symmetric.

    Parameters
    ----------
    A:
        Square SciPy sparse matrix to test.
    tol:
        Absolute tolerance for element-wise comparison of ``A`` and ``A.T``.

    Returns
    -------
    bool
        ``True`` if ``max(|A - A^T|) <= tol``, ``False`` otherwise.

    Raises
    ------
    ValueError
        If ``A`` is not square.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {A.shape}.")
    diff = A - A.T
    if not hasattr(diff, "data") or diff.nnz == 0:
        return True
    return float(np.abs(diff).max()) <= tol

def is_cg_compatible(
    A: sp.spmatrix,
    tol: float = 1e-10,
    max_condition_number: float = 1e10,
    verbose: bool = True,
) -> bool:
    """Check whether a SciPy sparse matrix is suitable for the Conjugate Gradient solver.

    CG requires the matrix to be **symmetric positive definite (SPD)** and
    reasonably well-conditioned:

    * **Symmetric**: ``max(|A - A^T|) <= tol``.
    * **Positive definite**: smallest eigenvalue > 0.
    * **Well-conditioned**: ``κ = λ_max / λ_min <= max_condition_number``.

    A large condition number (e.g. > 1e6) means CG will converge slowly or
    produce inaccurate results without a good preconditioner.
    """
    # 1) Symmetry check
    if not is_symmetric(A, tol=tol):
        if verbose:
            print("[CG check] FAILED: matrix is not symmetric.")
        return False

    # 2) Estimate smallest and largest eigenvalues
    try:
        lam_min = spla.eigsh(A, k=1, which="SM", return_eigenvectors=False)[0]
        lam_max = spla.eigsh(A, k=1, which="LM", return_eigenvectors=False)[0]
    except Exception as e:
        if verbose:
            print(f"[CG check] FAILED: eigenvalue estimation error: {e}")
        return False

    # 3) Positive definite check
    if lam_min <= 0:
        if verbose:
            print(f"[CG check] FAILED: not positive definite (λ_min={lam_min:.3e}).")
        return False

    # 4) Condition number check
    kappa = lam_max / lam_min
    if verbose:
        print(f"[CG check] λ_min={lam_min:.3e}, λ_max={lam_max:.3e}, κ={kappa:.3e}")

    if kappa > max_condition_number:
        if verbose:
            print(
                f"[CG check] WARNING: ill-conditioned matrix (κ={kappa:.3e} > {max_condition_number:.0e})."
                " CG may diverge or give inaccurate results. Consider a better preconditioner."
            )
        return False

    if verbose:
        print("[CG check] PASSED: matrix is SPD and well-conditioned.")
    return True


@timing_decorator
def solve_cg_cpu(
    A_cpu: sp.spmatrix,
    b_cpu: np.ndarray,
    tol: float = 1e-5,
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
    """
    x, info = spla.cg(A_cpu, b_cpu, rtol=tol)
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
