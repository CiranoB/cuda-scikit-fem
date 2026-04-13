from skfem import *
from skfem.helpers import dot, grad

import os

REFINED_TIMES: int = int(os.getenv("REFINED_TIMES", 3))
TOLERANCE: float = float(os.getenv("TOLERANCE", 1e-5))
HALF_PRECISION: bool = bool(int(os.getenv("HALF_PRECISION", "0")))

print("--------------------")
print("Ex 01, refined times: ", REFINED_TIMES)
print("Tolerance: ", TOLERANCE)
print("Using half precision" if HALF_PRECISION else "Using double precision")

# # enable additional mesh validity checks, sacrificing performance
# import logging
# logging.basicConfig(format='%(levelname)s %(asctime)s %(name)s %(message)s')
# logging.getLogger('skfem').setLevel(logging.DEBUG)

# create the mesh
m = MeshTri().refined(REFINED_TIMES)
# or, with your own points and cells:
# m = MeshTri(points, cells)
# or, load from file
# m = MeshTri.load("mesh.msh")

e = ElementTriP1()
basis = Basis(m, e)


@BilinearForm
def laplace(u, v, _):
    return dot(grad(u), grad(v))


@LinearForm
def rhs(v, _):
    return 1.0 * v


A = laplace.assemble(basis)
b = rhs.assemble(basis)

# enforce Dirichlet boundary conditions
A, b = enforce(A, b, D=m.boundary_nodes())

# solve -- can be anything that takes a sparse matrix and a right-hand side
x = solve(A, b)

def visualize():
    from skfem.visuals.matplotlib import plot
    return plot(m, x, shading='gouraud', colorbar=True)

# if __name__ == "__main__":
#     visualize().show()
