import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def is_symmetric(A: sp.spmatrix, tol: float = 1e-10) -> bool:
    """Check whether a sparse matrix is symmetric within ``tol``.

    Raises ``ValueError`` if ``A`` is not square.
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
    verbose: bool = True,
) -> bool:
    """Check whether a sparse matrix is suitable for CG.

    CG requires the matrix to be square, symmetric, and positive definite.
    Only the checks strictly required by the method are performed.

    ``max_condition_number`` is accepted for backward compatibility but does
    not affect the result.
    """
    if A.shape[0] != A.shape[1]:
        if verbose:
            print(f"[CG check] FAILED: matrix is not square (shape={A.shape}).")
        return False

    # 1) Symmetry check
    if not is_symmetric(A, tol=tol):
        if verbose:
            print("[CG check] FAILED: matrix is not symmetric.")
        return False

    # 2) Positive definiteness check via the smallest algebraic eigenvalue
    try:
        lam_min = spla.eigsh(A, k=1, which="SA", return_eigenvectors=False)[0]
    except Exception as e:
        if verbose:
            print(f"[CG check] FAILED: eigenvalue estimation error: {e}")
        return False

    # 3) Positive definite check
    if lam_min <= 0:
        if verbose:
            print(f"[CG check] FAILED: not positive definite (λ_min={lam_min:.3e}).")
        return False

    if verbose:
        print(f"[CG check] PASSED: matrix is symmetric positive definite (λ_min={lam_min:.3e}).")
    return True


def is_gmres_compatible(
    A: sp.spmatrix,
    verbose: bool = True,
) -> bool:
    """Check whether a sparse matrix is applicable for GMRES.

    GMRES requires a non-singular square matrix.  This function checks
    squareness and detects structural singularity via empty rows or columns.
    """
    # 1) GMRES is only defined for square systems
    if A.shape[0] != A.shape[1]:
        if verbose:
            print(f"[GMRES check] FAILED: matrix is not square (shape={A.shape}).")
        return False

    # 2) Cheap structural singularity check: any all-zero row or column means
    #    the system has no unique solution, making GMRES inapplicable.
    A_csr = A.tocsr() if not isinstance(A, sp.csr_matrix) else A

    if np.any(A_csr.getnnz(axis=1) == 0):
        if verbose:
            print("[GMRES check] FAILED: matrix has at least one all-zero row (structurally singular).")
        return False

    A_csc = A.tocsc()
    if np.any(A_csc.getnnz(axis=0) == 0):
        if verbose:
            print("[GMRES check] FAILED: matrix has at least one all-zero column (structurally singular).")
        return False

    if verbose:
        print("[GMRES check] PASSED: matrix passed structural checks and is applicable for GMRES.")

    return True
