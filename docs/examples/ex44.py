"""Wave equation."""
import os

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import splu
from skfem import *
from skfem.models import laplace, mass

INCREASE_REFINE_MESH = int(os.environ.get("INCREASE_REFINE_MESH", "1"))
SKIP_VISUALISATION = os.environ.get("SKIP_VISUALISATION", "0") == "1"


m = MeshLine().refined(6 + INCREASE_REFINE_MESH)
basis = Basis(m, ElementLineP1())

N = basis.N
L = laplace.assemble(basis)
M = mass.assemble(basis)
I = identity(N)
c = 1.

# reduction to first order system in time
A0 = bmat([[I, None],
           [None, M]], "csc")

B0 = bmat([[None, I],
           [-c ** 2 * L, None]], "csc")

# Crank-Nicolson
dt = .01
theta = .5
A = A0 + theta * B0 * dt
B = A0 - (1. - theta) * B0 * dt
backsolve = splu(A).solve

# timestepping
def evolve(t, u):
    while t < 1:
        t = t + dt
        u = backsolve(B @ u)
        yield t, u

# initial conditions
def bump(x):
    return np.exp(-100. * (x[0] - .5) ** 2)

U0 = basis.project(bump)
U = np.concatenate((U0, np.zeros(N)))


if __name__ == "__main__" and not SKIP_VISUALISATION:
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim

    fig, ax = plt.subplots()
    ix = np.argsort(m.p[0])
    line, = ax.plot(m.p[0, ix], U0[ix])

    def update(t_u):
        t, u = t_u
        u1, _ = np.split(u, [M.shape[0]])
        line.set_ydata(u1[ix])

    animation = anim.FuncAnimation(fig, update, evolve(0., U), interval=50)
    animation.save(Path(__file__).with_suffix(".gif"), "imagemagick")

