"""Discontinuous Galerkin method."""

from skfem import *
from skfem.helpers import grad, dot, jump
from skfem.models.poisson import laplace, unit_load
import os

INCREASE_REFINE_MESH = int(os.environ.get("INCREASE_REFINE_MESH", "1"))
SKIP_VISUALISATION = os.environ.get("SKIP_VISUALISATION", "0") == "1"

m = MeshTri.init_sqsymmetric().refined(1 + INCREASE_REFINE_MESH)
e = ElementTriDG(ElementTriP4())
alpha = 1e-3

ib = Basis(m, e)
bb = FacetBasis(m, e)
fb = InteriorFacetBasis(m, e, side=0) @ InteriorFacetBasis(m, e, side=1)


@BilinearForm
def dgform(u1, u2, v1, v2, w):
    ju, jv = u1 - u2, v1 - v2
    h = w.h
    n = w.n
    mu, mv = (
        0.5 * (dot(grad(u1), n) + dot(grad(u2), n)),
        0.5 * (dot(grad(v1), n) + dot(grad(v2), n)),
    )
    return ju * jv / (alpha * h) - mu * jv - mv * ju


@BilinearForm
def nitscheform(u, v, w):
    h = w.h
    n = w.n
    return u * v / (alpha * h) - dot(grad(u), n) * v - dot(grad(v), n) * u


A = laplace.assemble(ib)
b = unit_load.assemble(ib)
C = nitscheform.assemble(bb)
B = dgform.assemble(fb)

x = solve(A + B + C, b)

M, X = ib.refinterp(x, 4)

def visualize():
    from skfem.visuals.matplotlib import plot, draw
    ax = draw(M, boundaries_only=True)
    return plot(M, X, shading="gouraud", ax=ax, colorbar=True)

if __name__ == "__main__" and not SKIP_VISUALISATION:
    visualize().show()
