r"""Integral condition.

This short example demonstrates the implementation of an integral boundary
 condition

.. math::
   
   \int_\Gamma \nabla u \cdot \boldsymbol{n} \, \mathrm{d}s = 1

on a part of the boundary of the domain :math:`\Gamma \subset \partial \Omega`
 for the Laplace operator.  In this example, :math:`\Gamma` is the right
 boundary of the unit square and the solution satisfies :math:`u=0` on the
 bottom boundary and :math:`\nabla u \cdot \boldsymbol{n} = 0` on the rest of
 the boundaries.  The constraint is introduced via a Lagrange multiplier leading
 to a saddle point system.

"""
from skfem import *
from skfem.helpers import dot, grad
from skfem.models.poisson import laplace
import numpy as np
import scipy.sparse
import os

INCREASE_REFINE_MESH = int(os.environ.get("INCREASE_REFINE_MESH", "1"))
SKIP_VISUALISATION = os.environ.get("SKIP_VISUALISATION", "0") == "1"


m = MeshTri().refined(5 + INCREASE_REFINE_MESH).with_boundaries({"plate": lambda x: x[1] == 0.0})

e = ElementTriP1()

basis = Basis(m, e)
fbasis = FacetBasis(m, e)


@BilinearForm
def facetbilinf(u, v, w):
    n = w.n
    x = w.x
    return -dot(grad(u), n) * v * (x[0] == 1.0)


@LinearForm
def facetlinf(v, w):
    n = w.n
    x = w.x
    return -dot(grad(v), n) * (x[0] == 1.0)


A = laplace.assemble(basis)
B = facetbilinf.assemble(fbasis)
b = facetlinf.assemble(fbasis)

I = basis.complement_dofs(basis.get_dofs("plate"))


b = scipy.sparse.csr_matrix(b)

# create a block system with an extra Lagrange multiplier
K = scipy.sparse.bmat([[A + B, b.T], [b, None]], 'csr')
f = np.concatenate((basis.zeros(), -1.0 * np.ones(1)))

I = np.append(I, K.shape[0] - 1)

x = solve(*condense(K, f, I=I))

if __name__ == "__main__" and not SKIP_VISUALISATION:
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import plot, savefig
    plot(m, x[:-1], colorbar=True, shading='gouraud')
    savefig(splitext(argv[0])[0] + '_solution.png')
