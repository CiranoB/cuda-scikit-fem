"""This module contains utility functions such as convenient access to
SciPy linear solvers."""

from re import X
import sys
import inspect
import logging
import time
from typing import Optional, Union, Tuple, Callable, Dict

import numpy as np
from skfem.sparse_solvers import load_to_gpu, read_from_gpu, solve_cg_cpu, solve_cg_gpu, solve_gmres_gpu, get_jacobi_preconditioner, get_jacobi_preconditioner_cpu, get_ilu_preconditioner, get_block_jacobi_preconditioner, get_polynomial_preconditioner, solve_spsolve_cpu, solve_spsolve_gpu, solve_superlu_gpu
from skfem.solvers_aplicability import is_cg_compatible as check_cg_applicable, is_gmres_compatible as check_gmres_applicable
import scipy.sparse as sp
import scipy.sparse.csgraph as spg
import scipy.sparse.linalg as spl
from scipy.sparse.linalg import spsolve
from numpy import ndarray

import os

### CUDA ###
import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator
from cupyx.scipy.sparse import csr_matrix as cpx_csr
from cupyx.scipy.sparse.linalg import cg, gmres, cgs, minres
### CUDA ###

if "pyodide" in sys.modules:
    from scipy.sparse.base import spmatrix
else:
    from scipy.sparse import spmatrix

from skfem.assembly import asm, BilinearForm, LinearForm, DofsView
from skfem.assembly.basis import AbstractBasis
from skfem.element import ElementVector
from skfem.generic_utils import deprecated


logger = logging.getLogger(__name__)


# custom types for describing input and output values


Solution = Union[ndarray, Tuple[ndarray, ndarray]]
LinearSolver = Callable[..., ndarray]
EigenSolver = Callable[..., Tuple[ndarray, ndarray]]
LinearSystem = Union[spmatrix,
                     Tuple[spmatrix, ndarray],
                     Tuple[spmatrix, spmatrix]]
CondensedSystem = Union[LinearSystem,
                        Tuple[spmatrix, ndarray, ndarray],
                        Tuple[spmatrix, ndarray, ndarray, ndarray],
                        Tuple[spmatrix, spmatrix, ndarray, ndarray]]
DofsCollection = Union[ndarray, DofsView, Dict[str, DofsView]]


# preconditioners, e.g. for :func:`skfem.utils.solver_iter_krylov`


def build_pc_ilu(A: spmatrix,
                 drop_tol: Optional[float] = 1e-4,
                 fill_factor: Optional[float] = 20) -> spl.LinearOperator:
    """Incomplete LU preconditioner."""
    print("A size: ", A.size)
    P = spl.spilu(A.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
    M = spl.LinearOperator(A.shape, matvec=P.solve)
    return M


def build_pc_diag(A: spmatrix) -> spmatrix:
    """Diagonal preconditioner."""
    return sp.spdiags(1.0/A.diagonal(), 0, A.shape[0], A.shape[0])


# solvers for :func:`skfem.utils.solve`


def solver_eigen_scipy(**kwargs) -> EigenSolver:
    """Solve generalized eigenproblem using SciPy (ARPACK).

    Returns
    -------
    EigenSolver
        A solver function that can be passed to :func:`solve`.

    """
    params = {
        'sigma': 10,
        'k': 5,
    }
    params.update(kwargs)

    def solver(K, M, **solve_time_kwargs):
        params.update(solve_time_kwargs)
        from scipy.sparse.linalg import eigs
        return eigs(K, M=M, **params)

    return solver


def solver_eigen_scipy_sym(**kwargs) -> EigenSolver:
    """Solve symmetric generalized eigenproblem using SciPy (ARPACK).

    Returns
    -------
    EigenSolver
        A solver function that can be passed to :func:`solve`.

    """
    params = {
        'sigma': 10,
        'k': 5,
        'mode': 'normal',
    }
    params.update(kwargs)

    def solver(K, M, **solve_time_kwargs):
        params.update(solve_time_kwargs)
        from scipy.sparse.linalg import eigsh
        return eigsh(K, M=M, **params)

    return solver

TOLERANCE: float = float(os.getenv("TOLERANCE", 1e-5))


def _gpu_solver_enabled(name: str) -> bool:
    """Whether the GPU solver ``name`` is enabled via ``{name}_GPU_ENABLE``.

    GPU solvers are opt-in: nothing is computed on the GPU unless the matching
    environment variable is set to a truthy value (e.g. ``GMRES_GPU_ENABLE=1``).
    """
    return os.getenv(f"{name}_GPU_ENABLE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


 # Compare results
def compare(name, candidate, reference):
    diff = np.abs(candidate - reference)
    rel_err = diff / (np.abs(reference) + 1e-12)
    print(f"[{name}] max abs diff: {diff.max():.6e}, "
        f"mean abs diff: {diff.mean():.6e}, "
        f"max rel error: {rel_err.max():.6e}")
        
def solve_multiple_solver(A, b, x_cpu):
    # CPU CG solver
    # x_cg_cpu = solve_cg_cpu(A, b, TOLERANCE)
    # x_cg_cpu_jacobi = solve_cg_cpu(A, b, TOLERANCE, M=get_jacobi_preconditioner_cpu(A))

    # Load to GPU
    A_gpu, b_gpu = load_to_gpu(A,b)

    # x_spsolve_gpu = None
    # # Solvers
    # if not bool(os.getenv("DISABLE_SPSOLVE_GPU", False)):
    #     try:
    #         x_gpu = solve_spsolve_gpu(A_gpu, b_gpu)
    #         x_spsolve_gpu = read_from_gpu(x_gpu)
    #         cp.get_default_memory_pool().free_all_blocks()
    #         del x_gpu
    #     except Exception:
    #         os.environ["DISABLE_SPSOLVE_GPU"] = "1"
    #         print("Skipping SPSOLVE GPU due memory constrains")

    # x_superlu_gpu = None
    # if not bool(os.getenv("DISABLE_SUPERLU_GPU", False)):
    #     try:
    #         x_gpu = solve_superlu_gpu(A_gpu, b_gpu)
    #         x_superlu_gpu = read_from_gpu(x_gpu)
    #         cp.get_default_memory_pool().free_all_blocks()
    #         del x_gpu
    #     except Exception:
    #         os.environ["DISABLE_SUPERLU_GPU"] = "1"
    #         print("Skipping SPSOLVE GPU due memory constrains")

    x_gpu = solve_cg_gpu(A_gpu, b_gpu, TOLERANCE)
    x_cg_no_prec = read_from_gpu(x_gpu)
    cp.get_default_memory_pool().free_all_blocks()
    del x_gpu

    x_gpu = solve_cg_gpu(A_gpu, b_gpu, TOLERANCE, M=get_jacobi_preconditioner(A_gpu))
    x_cg_jacobi = read_from_gpu(x_gpu)
    cp.get_default_memory_pool().free_all_blocks()
    del x_gpu


   

    # if x_spsolve_gpu is not None:
    #     compare("spsolve GPU vs CPU spsolve", x_spsolve_gpu, x_cpu)
    # else: 
    #     print("Skipping SPSOLVE GPU comparison due memory constrains")

    # if x_superlu_gpu is not None:
    #     compare("SuperLU GPU vs CPU spsolve", x_superlu_gpu, x_cpu)
    # else: 
    #     print("Skipping SUPERLU GPU comparison due memory constrains")

    # compare("CG CPU vs CPU spsolve", x_cg_cpu, x_cpu)
    # compare("CG (Jacobi precond) CPU vs CPU spsolve", x_cg_cpu_jacobi, x_cpu)
    compare("CG (no precond) GPU vs CPU spsolve", x_cg_no_prec, x_cpu)
    compare("CG (Jacobi precond) GPU vs CPU spsolve", x_cg_jacobi, x_cpu)
            
    # Cleanup
    del A_gpu, b_gpu
    cp.get_default_memory_pool().free_all_blocks()

def solve_gmres_solver(A, b, x_cpu):
    # Load to GPU
    A_gpu, b_gpu = load_to_gpu(A,b)

    x_spsolve_gpu = None

    x_gpu = solve_gmres_gpu(A_gpu,
                            b_gpu,
                            TOLERANCE,
                            get_ilu_preconditioner(A_gpu)
    )
    x_gpu_gmres = read_from_gpu(x_gpu)
    cp.get_default_memory_pool().free_all_blocks()
    # del x_gpu

    compare("GMRES CPU vs CPU spsolve", x_gpu_gmres, x_cpu)

    # Cleanup
    del A_gpu, b_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return x_gpu_gmres


# Registry of opt-in GPU solvers keyed by the prefix of their enable env var.
# Each entry solves ``A x = b`` on the GPU and is only run when
# ``{NAME}_GPU_ENABLE`` is truthy (see :func:`_gpu_solver_enabled`).
_GPU_SOLVERS: Dict[str, Callable] = {
    "SPSOLVE": lambda A_gpu, b_gpu: solve_spsolve_gpu(A_gpu, b_gpu),
    "SUPERLU": lambda A_gpu, b_gpu: solve_superlu_gpu(A_gpu, b_gpu),
    "CG": lambda A_gpu, b_gpu: solve_cg_gpu(
        A_gpu, b_gpu, TOLERANCE, M=get_jacobi_preconditioner(A_gpu)),
    "GMRES": lambda A_gpu, b_gpu: solve_gmres_gpu(
        A_gpu, b_gpu, TOLERANCE, M=get_ilu_preconditioner(A_gpu)),
}


def run_enabled_gpu_solvers(A, b, x_cpu):
    """Run each GPU solver whose ``{NAME}_GPU_ENABLE`` env var is truthy.

    The system is uploaded to the GPU once and shared across the enabled
    solvers.  Every enabled solver is timed (via the decorators in
    ``sparse_solvers``) and its result compared against the CPU reference
    ``x_cpu``.  Returns immediately without touching the GPU when nothing is
    enabled.
    """
    enabled = [name for name in _GPU_SOLVERS if _gpu_solver_enabled(name)]
    if not enabled:
        return

    A_gpu, b_gpu = load_to_gpu(A, b)
    try:
        for name in enabled:
            x_gpu = _GPU_SOLVERS[name](A_gpu, b_gpu)
            compare(f"{name} GPU vs CPU spsolve", read_from_gpu(x_gpu), x_cpu)
            del x_gpu
            cp.get_default_memory_pool().free_all_blocks()
    finally:
        del A_gpu, b_gpu
        cp.get_default_memory_pool().free_all_blocks()


# ---------------------------------------------------------------------------
# CG preconditioner benchmark (opt-in via CG_PRECOND_BENCHMARK)
# ---------------------------------------------------------------------------

# GPU CG cases compared against the CPU spsolve reference, in report order.
_CG_BENCH_CASES = ("none", "jacobi", "ilu0", "block_jacobi", "polynomial")


class _CGBudgetExceeded(Exception):
    """Internal signal that a CG case exceeded its per-case time budget."""

# Only the first eligible linear solve per process is benchmarked so that
# time-stepping / nonlinear examples are not measured on every internal solve.
# The subprocess-per-run harness resets this flag naturally.
_cg_bench_done = False


def _cg_bench_enabled() -> bool:
    """Whether the 5-case CG preconditioner benchmark is opted in."""
    return os.getenv("CG_PRECOND_BENCHMARK", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _cg_bench_disabled_cases() -> set[str]:
    """Return benchmark cases disabled for the current exercise sweep."""
    return {
        case.strip()
        for case in os.getenv("CG_BENCHMARK_DISABLED_CASES", "").split(",")
        if case.strip()
    }


def _make_cg_preconditioner(case: str, A_gpu):
    """Build the GPU preconditioner for a benchmark ``case`` (``None`` if any)."""
    if case == "none":
        return None
    if case == "jacobi":
        return get_jacobi_preconditioner(A_gpu)
    if case == "ilu0":
        return get_ilu_preconditioner(A_gpu)
    if case == "block_jacobi":
        return get_block_jacobi_preconditioner(
            A_gpu, block_size=int(os.getenv("CG_BLOCK_SIZE", "8")))
    if case == "polynomial":
        return get_polynomial_preconditioner(
            A_gpu, degree=int(os.getenv("CG_POLY_DEGREE", "3")))
    raise ValueError(f"unknown CG benchmark case: {case}")


def run_cg_preconditioner_benchmark(A, b, x_cpu, spsolve_cpu_s=-1.0):
    """Benchmark GPU CG with several preconditioners against CPU spsolve.

    Opt-in via ``CG_PRECOND_BENCHMARK``.  Uploads the condensed system once and,
    for each preconditioner in :data:`_CG_BENCH_CASES`, times the end-to-end GPU
    CG path (host->device transfer, preconditioner build, solve, device->host
    readback) and reports convergence, the 2-norm relative residual, and the
    difference from the CPU reference on machine-parseable ``[cgcase]`` lines.
    Only the first eligible solve per process is measured.
    """
    global _cg_bench_done
    if _cg_bench_done or not _cg_bench_enabled():
        return

    ndofs = A.shape[0]
    if ndofs < int(os.getenv("CG_BENCH_MIN_DOFS", "1")):
        return
    _cg_bench_done = True

    tol = TOLERANCE
    maxiter = int(os.getenv("CG_MAXITER", "5000"))
    bnorm = float(np.linalg.norm(b)) or 1.0
    supports_cb = "callback" in inspect.signature(cg).parameters

    # Warm up the CUDA context so ``transfer_s`` measures the host->device copy
    # rather than one-time context initialization, which a real application
    # pays once at start-up instead of on every solve.
    _warm = cp.zeros(1) + 1
    cp.cuda.Stream.null.synchronize()
    del _warm

    t0 = time.perf_counter()
    A_gpu, b_gpu = load_to_gpu(A, b)
    cp.cuda.Stream.null.synchronize()
    transfer_s = time.perf_counter() - t0

    print(f"[cgbench] ndofs={ndofs} nnz={int(A.nnz)} "
          f"spsolve_cpu_s={spsolve_cpu_s:.6f} transfer_s={transfer_s:.6f} "
          f"tol={tol:g} maxiter={maxiter}", flush=True)

    budget = float(os.getenv("CG_CASE_TIMEOUT_S", "60"))
    disabled_cases = _cg_bench_disabled_cases()
    try:
        for case in _CG_BENCH_CASES:
            if case in disabled_cases:
                print(f"[cgcase] case={case} status=skipped", flush=True)
                continue
            try:
                t0 = time.perf_counter()
                M = _make_cg_preconditioner(case, A_gpu)
                cp.cuda.Stream.null.synchronize()
                precond_s = time.perf_counter() - t0

                state = {"iters": 0, "last_x": None, "start": 0.0}

                def _cb(xk, _s=state):
                    _s["iters"] += 1
                    _s["last_x"] = xk
                    if time.perf_counter() - _s["start"] > budget:
                        raise _CGBudgetExceeded()

                cg_kwargs = {"rtol": tol, "maxiter": maxiter, "M": M}
                if supports_cb:
                    cg_kwargs["callback"] = _cb

                status = "converged"
                state["start"] = time.perf_counter()
                t0 = state["start"]
                try:
                    x_gpu, info = cg(A_gpu, b_gpu, **cg_kwargs)
                    if info != 0:
                        status = "maxiter"
                except _CGBudgetExceeded:
                    x_gpu = state["last_x"]
                    status = "timeout"
                cp.cuda.Stream.null.synchronize()
                solve_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                x_np = cp.asnumpy(x_gpu)
                readback_s = time.perf_counter() - t0

                total_s = transfer_s + precond_s + solve_s + readback_s
                diff = np.abs(x_np - x_cpu)
                relres = float(np.linalg.norm(A @ x_np - b) / bnorm)
                print(f"[cgcase] case={case} status={status} "
                      f"precond_s={precond_s:.6f} solve_s={solve_s:.6f} "
                      f"readback_s={readback_s:.6f} total_s={total_s:.6f} "
                      f"converged={int(status == 'converged')} "
                      f"iters={state['iters'] if supports_cb else -1} "
                      f"relres={relres:.3e} max_abs_diff={diff.max():.3e} "
                      f"mean_abs_diff={diff.mean():.3e}", flush=True)

                del M, x_gpu
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as exc:
                print(f"[cgcase] case={case} status=error "
                      f"error={type(exc).__name__}", flush=True)
                cp.get_default_memory_pool().free_all_blocks()
    finally:
        del A_gpu, b_gpu
        cp.get_default_memory_pool().free_all_blocks()


def solver_direct_scipy(**kwargs):
    def solver(A: sp.spmatrix, b: np.ndarray, **solve_time_kwargs):
        local_kwargs = kwargs.copy()
        local_kwargs.update(solve_time_kwargs)

        t_spsolve_start = time.perf_counter()
        x = solve_spsolve_cpu(A, b)
        spsolve_cpu_s = time.perf_counter() - t_spsolve_start

        run_enabled_gpu_solvers(A, b, x)
        run_cg_preconditioner_benchmark(A, b, x, spsolve_cpu_s=spsolve_cpu_s)

        return x

    return solver


def solver_iter_krylov(krylov: Optional[LinearSolver] = spl.cg,
                       verbose: Optional[bool] = False,
                       **kwargs) -> LinearSolver:
    """Krylov-subspace iterative linear solver.

    Parameters
    ----------
    krylov
        A Krylov iterative linear solver, like, and by default,
        :func:`scipy.sparse.linalg.cg`
    verbose
        If True, print the norm of the iterate.

    Any remaining keyword arguments are passed on to the solver, in particular
    tol and atol, the tolerances, maxiter, and M, the preconditioner.  If the
    last is omitted, a diagonal preconditioner is supplied using
    :func:`skfem.utils.build_pc_diag`.

    Returns
    -------
    LinearSolver
        A solver function that can be passed to :func:`solve`.

    """
    def callback(x):
        if verbose:
            print(np.linalg.norm(x))

    def solver(A, b, **solve_time_kwargs):
        kwargs.update(solve_time_kwargs)
        if 'M' not in kwargs:
            kwargs['M'] = build_pc_diag(A)
        sol, info = krylov(A, b, **{'callback': callback, **kwargs})
        if info > 0:
            logger.warning("Iterative solver did not converge.")
        elif info == 0 and verbose:
            print(f"{krylov.__name__} converged to "
                  + f"tol={kwargs.get('tol', 'default')} and "
                  + f"atol={kwargs.get('atol', 'default')}")
        return sol

    return solver


def solver_iter_pcg(**kwargs) -> LinearSolver:
    """Conjugate gradient solver, specialized from solver_iter_krylov"""
    return solver_iter_krylov(**kwargs)


def solver_iter_cg(**kwargs):
    """Pure Python conjugate gradient solver (for old scipy versions)."""

    def solver(A, b, **solve_time_kwargs):
        kwargs.update(solve_time_kwargs)
        maxiters = kwargs['maxiters'] if 'maxiters' in kwargs else 500
        tol = kwargs['tol'] if 'tol' in kwargs else 1e-10
        x = b
        r = b - A.dot(x)
        p = r
        rsold = np.dot(r, r)
        for k in range(maxiters):
            Ap = A.dot(p)
            alpha = rsold / np.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = np.dot(r, r)
            if np.sqrt(rsnew) < tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        if k == maxiters:
            logger.warning("Iterative solver did not converge.")
        return x

    return solver


# solve and condense

def solve_eigen(A: spmatrix,
                M: spmatrix,
                x: Optional[ndarray] = None,
                I: Optional[ndarray] = None,
                solver: Optional[EigenSolver] = None,
                **kwargs) -> Tuple[ndarray, ndarray]:

    if solver is None:
        solver = solver_eigen_scipy(**kwargs)

    if x is not None and I is not None:
        L, X = solver(A, M, **kwargs)
        y = np.tile(x.copy()[:, None], (1, X.shape[1]))
        if isinstance(I, tuple):
            np.add.at(y, I[0], np.array([I[1](x) for x in X.T]).T)
        else:
            y[I] = X
        return L, y
    return solver(A, M, **kwargs)


def solve_linear(A: spmatrix,
                 b: ndarray,
                 x: Optional[ndarray] = None,
                 I: Optional[ndarray] = None,
                 solver: Optional[LinearSolver] = None,
                 **kwargs) -> ndarray:

    if solver is None:
        solver = solver_direct_scipy(**kwargs)

    if x is not None and I is not None:
        y = x.copy()
        if isinstance(I, tuple):
            np.add.at(y, I[0], I[1](solver(A, b, **kwargs)))
        else:
            y[I] = solver(A, b, **kwargs)
        return y
    return solver(A, b, **kwargs)


def solve(A: spmatrix,
          b: Union[ndarray, spmatrix],
          x: Optional[ndarray] = None,
          I: Optional[ndarray] = None,
          solver: Optional[Union[LinearSolver, EigenSolver]] = None,
          **kwargs) -> Solution:
    """Solve a linear system or a generalized eigenvalue problem.

    The remaining keyword arguments are passed to the solver.

    Parameters
    ----------
    A
        The system matrix
    b
        The right hand side vector or the mass matrix of a generalized
        eigenvalue problem.
    solver
        Choose one of the following solvers:
        :func:`skfem.utils.solver_direct_scipy` (default),
        :func:`skfem.utils.solver_eigen_scipy` (default),
        :func:`skfem.utils.solver_iter_pcg`,
        :func:`skfem.utils.solver_iter_krylov`.

    """
    logger.info("Solving linear system, shape={}.".format(A.shape))
    if isinstance(b, spmatrix):
        out = solve_eigen(A, b, x, I, solver, **kwargs)  # type: ignore
    elif isinstance(b, ndarray):
        out = solve_linear(A, b, x, I, solver, **kwargs)  # type: ignore
    else:
        raise NotImplementedError("Provided argument types not supported")
    logger.info("Solving done.")
    return out


def _flatten_dofs(S: Optional[DofsCollection]) -> Optional[ndarray]:
    if S is None:
        return None
    if isinstance(S, ndarray):
        return S
    elif isinstance(S, DofsView):
        return S.flatten()
    elif isinstance(S, dict):
        def _flatten_helper(S, key):
            if key in S and isinstance(S[key], DofsView):
                return S[key].flatten()
            raise NotImplementedError
        return np.unique(
            np.concatenate([_flatten_helper(S, key) for key in S])
        )
    raise NotImplementedError("Unable to flatten the given set of DOFs.")


def _init_bc(A: spmatrix,
             b: Optional[Union[ndarray, spmatrix]] = None,
             x: Optional[ndarray] = None,
             I: Optional[DofsCollection] = None,
             D: Optional[DofsCollection] = None) -> Tuple[Optional[ndarray],
                                                          ndarray,
                                                          ndarray,
                                                          ndarray]:

    D = _flatten_dofs(D)
    I = _flatten_dofs(I)

    if I is None and D is None:
        raise Exception("Either I or D must be given!")
    elif I is None and D is not None:
        I = np.setdiff1d(np.arange(A.shape[0], dtype=np.int32), D)
    elif D is None and I is not None:
        D = np.setdiff1d(np.arange(A.shape[0], dtype=np.int32), I)
    else:
        raise Exception("Give only I or only D!")

    assert isinstance(I, ndarray)
    assert isinstance(D, ndarray)

    if x is None:
        x = np.zeros(A.shape[0], dtype=A.dtype)
    elif b is None:
        b = np.zeros_like(x)

    return b, x, I, D


def enforce(A: spmatrix,
            b: Optional[Union[ndarray, spmatrix]] = None,
            x: Optional[ndarray] = None,
            I: Optional[DofsCollection] = None,
            D: Optional[DofsCollection] = None,
            diag: float = 1.,
            overwrite: bool = False) -> LinearSystem:
    r"""Enforce degrees-of-freedom of a linear system.

    An alternative to :func:`~skfem.utils.condense` which sets the matrix
    diagonals to one and right-hand side vector to the enforced
    degree-of-freedom value.

    .. note::

        The original system is both returned
        (for compatibility with :func:`skfem.utils.solve`) and optionally (if
        `overwrite`) modified (for performance).

    Parameters
    ----------
    A
        The system matrix
    b
        Optionally, the right hand side vector.
    x
        The values of the enforced degrees-of-freedom. If not given, assumed
        to be zero.
    I
        Specify either this or ``D``: The set of degree-of-freedom indices to
        solve for.
    D
        Specify either this or ``I``: The set of degree-of-freedom indices to
        enforce (rows/diagonal set to zero/one).
    overwrite
        Optionally, the original system is both modified (for performance) and
        returned (for compatibility with :func:`skfem.utils.solve`).  By
        default, ``False``.

    Returns
    -------
    LinearSystem
        A linear system with the enforced rows/diagonals set to zero/one.

    """
    b, x, I, D = _init_bc(A, b, x, I, D)

    Aout = A if overwrite else A.copy()

    # set rows on lhs to zero
    start = Aout.indptr[D]
    stop = Aout.indptr[D + 1]
    count = stop - start
    idx = np.ones(count.sum(), dtype=np.int32)
    idx[np.cumsum(count)[:-1]] -= count[:-1]
    idx = np.repeat(start, count) + np.cumsum(idx) - 1
    Aout.data[idx] = 0.

    # set diagonal value
    d = Aout.diagonal()
    d[D] = diag
    Aout.setdiag(d)

    if b is not None:
        if isinstance(b, spmatrix):
            # mass matrix (eigen- or initial value problem)
            bout = enforce(b, D=D, diag=0., overwrite=overwrite)
        else:
            # set rhs to the given value
            bout = b if overwrite else b.copy()
            bout[D] = x[D]
        return Aout, bout

    return Aout


def penalize(A: spmatrix,
             b: Optional[Union[ndarray, spmatrix]] = None,
             x: Optional[ndarray] = None,
             I: Optional[DofsCollection] = None,
             D: Optional[DofsCollection] = None,
             epsilon: Optional[float] = None,
             overwrite: bool = False) -> LinearSystem:
    r"""Penalize degrees-of-freedom of a linear system.

    Parameters
    ----------
    A
        The system matrix
    b
        Optionally, the right hand side vector.
    x
        The values of the penalized degrees-of-freedom. If not given, assumed
        to be zero.
    I
        Specify either this or ``D``: The set of degree-of-freedom indices to
        solve for.
    D
        Specify either this or ``I``: The set of degree-of-freedom indices to
        enforce (rows/diagonal set to zero/one).
    epsilon
        Very small value, the reciprocal of which penalizes deviations from
        the Dirichlet condition
    overwrite
        Optionally, the original system is both modified (for performance) and
        returned (for compatibility with :func:`skfem.utils.solve`).  By
        default, ``False``.

    Returns
    -------
    LinearSystem
        A linear system with the penalized diagonal and RHS entries set to
        very large values, 1/epsilon and x/epsilon, respectively.

    """
    b, x, I, D = _init_bc(A, b, x, I, D)

    Aout = A if overwrite else A.copy()

    d = Aout.diagonal()
    if epsilon is None:
        epsilon = 1e-10 / np.linalg.norm(d[D], np.inf).astype(float)
    d[D] = 1. / epsilon
    Aout.setdiag(d)

    if b is None:
        return Aout

    bout = b if overwrite else b.copy()
    # Nothing needs doing for mass matrix, but RHS vector needs penalty factor
    if not isinstance(b, spmatrix):
        bout[D] = x[D] / epsilon
    return Aout, bout


def condense(A: spmatrix,
             b: Optional[Union[ndarray, spmatrix]] = None,
             x: Optional[ndarray] = None,
             I: Optional[DofsCollection] = None,
             D: Optional[DofsCollection] = None,
             expand: bool = True) -> CondensedSystem:
    r"""Eliminate degrees-of-freedom from a linear system.

    The user should provide the linear system ``A`` and ``b``
    and either the set of DOFs to eliminate (``D``) or the set
    of DOFs to keep (``I``).  Optionally, nonzero values for
    the eliminated DOFs can be supplied via ``x``.

    .. note::

        Supports also generalized eigenvalue problems
        where ``b`` is a matrix.

    Example
    -------

    Suppose that the solution vector :math:`x` can be
    split as

    .. math::

       x = \begin{bmatrix}
           x_I\\
           x_D
       \end{bmatrix}

    where :math:`x_D` are known and :math:`x_I` are unknown.  This allows
    splitting the linear system as

    .. math::

       \begin{bmatrix}
           A_{II} & A_{ID}\\
           A_{DI} & A_{DD}
       \end{bmatrix}
       \begin{bmatrix}
           x_I\\
           x_D
       \end{bmatrix}
       =
       \begin{bmatrix}
           b_I\\
           b_D
       \end{bmatrix}

    which leads to the condensed system

    .. math::

       A_{II} x_I = b_I - A_{ID} x_D.


    As an example, let us assemble the matrix :math:`A` and the vector
    :math:`b` corresponding to the Poisson equation :math:`-\Delta u = 1`.

    .. doctest::

       >>> import skfem as fem
       >>> from cudaskfem.models.poisson import laplace, unit_load
       >>> m = fem.MeshTri().refined(2)
       >>> basis = fem.CellBasis(m, fem.ElementTriP1())
       >>> A = laplace.assemble(basis)
       >>> b = unit_load.assemble(basis)

    The condensed system is obtained with :func:`skfem.utils.condense`.  Below
    we provide the DOFs to eliminate via the keyword argument
    ``D``.

    .. doctest::

       >>> AII, bI, xI, I = fem.condense(A, b, D=m.boundary_nodes())
       >>> AII.toarray()
       array([[ 4.,  0.,  0.,  0., -1., -1., -1., -1.,  0.],
              [ 0.,  4.,  0.,  0., -1.,  0., -1.,  0.,  0.],
              [ 0.,  0.,  4.,  0.,  0., -1.,  0., -1.,  0.],
              [ 0.,  0.,  0.,  4., -1., -1.,  0.,  0.,  0.],
              [-1., -1.,  0., -1.,  4.,  0.,  0.,  0.,  0.],
              [-1.,  0., -1., -1.,  0.,  4.,  0.,  0.,  0.],
              [-1., -1.,  0.,  0.,  0.,  0.,  4.,  0., -1.],
              [-1.,  0., -1.,  0.,  0.,  0.,  0.,  4., -1.],
              [ 0.,  0.,  0.,  0.,  0.,  0., -1., -1.,  4.]])
        >>> bI
        array([0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625,
               0.0625])

    By default, the eliminated DOFs are set to zero.
    Different values can be provided through the keyword argument ``x``;
    see :ref:`ex14`.

    Parameters
    ----------
    A
        The system matrix
    b
        The right hand side vector, or zero if x is given, or the mass matrix
        for generalized eigenvalue problems.
    x
        The values of the condensed degrees-of-freedom. If not given, assumed
        to be zero.
    I
        The set of degree-of-freedom indices to include.
    D
        The set of degree-of-freedom indices to dismiss.
    expand
        If ``True`` (default), returns also `x` and `I`. As a consequence,
        :func:`skfem.utils.solve` will expand the solution vector
        automatically.

    Returns
    -------
    CondensedSystem
        The condensed linear system and (optionally) information about
        the boundary values.

    """
    b, x, I, D = _init_bc(A, b, x, I, D)

    ret_value: CondensedSystem = (None,)

    if b is None:
        ret_value = (A[I][:, I],)
    else:
        if isinstance(b, spmatrix):
            # generalized eigenvalue problem: don't modify rhs
            Aout = A[I][:, I]
            bout = b[I][:, I]
        elif isinstance(b, ndarray):
            Aout = A[I][:, I]
            bout = b[I] - A[I][:, D] @ x[D]
        else:
            raise Exception("Type of second arg not supported.")
        ret_value = (Aout, bout)

    if expand:
        ret_value += (x, I)

    return ret_value if len(ret_value) > 1 else ret_value[0]


def mpc(A: spmatrix,
        b: ndarray,
        S: Optional[ndarray] = None,
        M: Optional[ndarray] = None,
        T: Optional[spmatrix] = None,
        g: Optional[ndarray] = None) -> CondensedSystem:
    """Apply a multipoint constraint on the linear system.

    Parameters
    ----------
    A
    b
        The linear system to constrain.
    S
    M
    T
    g
        The constraint is of the form `x[S] = T @ x[M] + g`.

    """
    if M is None:
        M = np.array([], dtype=np.int32)
    if S is None:
        S = np.array([], dtype=np.int32)

    U = np.setdiff1d(np.arange(A.shape[0], dtype=np.int32),
                     np.concatenate((M, S)))

    if T is None:
        T = sp.eye(len(S), len(M))
    if g is None:
        g = np.zeros(len(S))

    if T.shape[0] != len(S) or T.shape[1] != len(M) or len(g) != len(S):
        raise ValueError("Inputs to mpc have incompatible shapes.")

    B = bmat([
        [
            A[U][:, U],
            A[U][:, M] + A[U][:, S] @ T,
        ],
        [
            A[M][:, U],
            A[M][:, M] + A[M][:, S] @ T,
        ]], 'csr')
    y = np.concatenate((b[U] - A[U][:, S] @ g,
                        b[M] - A[M][:, S] @ g))

    return (
        B,
        y,
        np.zeros_like(b, dtype=B.dtype),
        (
            np.concatenate((U, M, S)),
            lambda x: np.concatenate((x, T @ x[len(U):] + g)),
        )
    )


# additional utilities


def bmat(blocks, *args, **kwargs):
    """A variant of scipy bmat which adds block indices to out.blocks."""
    m = len(blocks)
    n = len(blocks[0])

    # turn COOData into scipy/numpy
    blocks = [[coo.todefault() if hasattr(coo, 'todefault') else coo
               for coo in row] for row in blocks]

    sizes = []
    diff = 0

    for j in range(n - 1):
        for i in range(m):
            if blocks[i][j] is None:
                continue
            else:
                if len(blocks[i][j].shape) == 1:
                    sizes.append(blocks[i][j].shape[0] + diff)
                else:
                    sizes.append(blocks[i][j].shape[1] + diff)
                diff += sizes[-1]
                break

    mat = sp.bmat(blocks, *args, **kwargs)
    mat.blocks = sizes  # add block sizes as an attribute

    return mat


def rcm(A: spmatrix,
        b: ndarray) -> Tuple[spmatrix, ndarray, ndarray]:
    """Reverse Cuthill-McKee ordering."""
    p = spg.reverse_cuthill_mckee(A, symmetric_mode=False)
    return A[p].T[p].T, b[p], p


def adaptive_theta(est, theta=0.5, max=None):
    """For choosing which elements to refine in an adaptive strategy."""
    if max is None:
        return np.nonzero(theta * np.max(est) < est)[0].astype(np.int32)
    else:
        return np.nonzero(theta * max < est)[0].astype(np.int32)


@deprecated("Basis.project")
def projection(fun,
               basis_to: Optional[AbstractBasis] = None,
               basis_from: Optional[AbstractBasis] = None,
               diff: Optional[int] = None,
               I: Optional[ndarray] = None,
               expand: bool = False) -> ndarray:

    @BilinearForm
    def mass(u, v, w):
        from skfem.helpers import dot, ddot
        p = 0
        if len(u.shape) == 2:
            p = u * v
        elif len(u.shape) == 3:
            p = dot(u, v)
        elif len(u.shape) == 4:
            p = ddot(u, v)
        return p

    if isinstance(fun, LinearForm):
        funv = fun
    else:
        @LinearForm
        def funv(v, w):
            p = fun(w.x) * v
            return sum(p) if isinstance(basis_to.elem, ElementVector) else p

    @BilinearForm
    def deriv(u, v, w):
        from skfem.helpers import grad
        du = grad(u)
        return du[diff] * v

    M = asm(mass, basis_to)

    if not isinstance(fun, ndarray):
        f = asm(funv, basis_to)
    else:
        if diff is not None:
            f = asm(deriv, basis_from, basis_to) @ fun
        else:
            f = asm(mass, basis_from, basis_to) @ fun

    if I is not None:
        return solve_linear(*condense(M, f, I=I, expand=expand))

    return solve_linear(M, f)


@deprecated("Basis.project (will be removed in the next release)")
def project(fun,
            basis_from: Optional[AbstractBasis] = None,
            basis_to: Optional[AbstractBasis] = None,
            diff: Optional[int] = None,
            I: Optional[ndarray] = None,
            expand: bool = False) -> ndarray:
    return projection(
        fun,
        basis_to=basis_to,
        basis_from=basis_from,
        diff=diff,
        I=I,
        expand=expand,
    )
