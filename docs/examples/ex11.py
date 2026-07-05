r"""Linear elasticity.

This example solves the linear elasticity problem using trilinear elements.

"""
import os

import numpy as np
from skfem import *
from skfem.helpers import ddot, sym_grad, eye, trace
from skfem.models.elasticity import lame_parameters

INCREASE_REFINE_MESH = int(os.environ.get("INCREASE_REFINE_MESH", "1"))
SKIP_VISUALISATION = os.environ.get("SKIP_VISUALISATION", "0") == "1"


m = MeshHex().refined(3 + INCREASE_REFINE_MESH).with_defaults()
e = ElementVector(ElementHex1())
basis = Basis(m, e, intorder=3)

# calculate Lamé parameters from Young's modulus and Poisson ratio
lam, mu = lame_parameters(1e3, 0.3)


def C(T):
    return 2. * mu * T + lam * eye(trace(T), T.shape[0])


@BilinearForm
def stiffness(u, v, w):
    return ddot(C(sym_grad(u)), sym_grad(v))


K = stiffness.assemble(basis)

u = basis.zeros()
u[basis.get_dofs('right').nodal['u^1']] = 0.3

u = solve(*condense(K, x=u, D=basis.get_dofs({'left', 'right'})))

sf = 1.0
m = m.translated(sf * u[basis.nodal_dofs])

if __name__ == "__main__" and not SKIP_VISUALISATION:
    from os.path import splitext
    from sys import argv

    # save to VTK for visualization in Paraview
    m.save(splitext(argv[0])[0] + '.vtk')
