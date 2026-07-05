"""One-dimensional Poisson."""

import os

import numpy as np
from skfem import *
from skfem.models.poisson import laplace, unit_load

INCREASE_REFINE_MESH = int(os.environ.get("INCREASE_REFINE_MESH", "1"))
SKIP_VISUALISATION = os.environ.get("SKIP_VISUALISATION", "0") == "1"

m = MeshLine(np.linspace(0, 1, (10 - 1) * INCREASE_REFINE_MESH + 1))

e = ElementLineP1()
basis = Basis(m, e)

A = asm(laplace, basis)
b = asm(unit_load, basis)

x = solve(*condense(A, b, D=basis.get_dofs()))

if __name__ == "__main__" and not SKIP_VISUALISATION:
    from skfem.visuals.matplotlib import plot, show
    plot(m, x)
    show()
