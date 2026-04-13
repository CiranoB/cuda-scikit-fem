r"""Simply supported Euler-Bernoulli beam.

This example solves the Euler-Bernoulli beam equation

.. math::
   EI w'''' = q \quad \text{in } (0, L),

for a simply supported beam under a uniformly distributed transverse load.
The essential boundary conditions are the vanishing deflection at both ends,

.. math::
   w(0) = w(L) = 0,

whereas the vanishing bending moment,

.. math::
   M(0) = EI w''(0) = 0, \quad M(L) = EI w''(L) = 0,

is enforced naturally by the weak formulation.

The problem is discretized using cubic Hermite elements, which are well suited
for fourth-order beam models.  For a constant load the analytical solution is

.. math::
   w(x) = \frac{q x}{24 EI}\left(L^3 - 2 L x^2 + x^3\right),

and the maximum deflection occurs at midspan,

.. math::
   w\left(\frac{L}{2}\right) = \frac{5 q L^4}{384 EI}.

"""

import numpy as np

from skfem import *


length = 4.0               # m
youngs_modulus = 210e9     # Pa
second_moment = 8.33e-6    # m^4
distributed_load = -12e3   # N/m (downwards)

mesh = MeshLine(np.linspace(0.0, length, 33)).with_boundaries({
    "left": lambda x: x[0] == 0.0,
    "right": lambda x: x[0] == length,
}).refined(5)
basis = Basis(mesh, ElementLineHermite())


@BilinearForm
def bilinf(u, v, w):
    from skfem.helpers import dd, ddot
    return youngs_modulus * second_moment * ddot(dd(u), dd(v))  # type: ignore[arg-type]


@LinearForm
def linf(v, w):
    return distributed_load * v


A = asm(bilinf, basis)
f = asm(linf, basis)

# Simply supported beam: constrain only the deflection DOFs at both ends.
D = basis.get_dofs({"left", "right"}).all("u")

x = np.asarray(solve(*condense(A, f, D=D)))


def analytical_solution(xcoord: np.ndarray) -> np.ndarray:
    return (distributed_load
            * xcoord
            * (length ** 3 - 2.0 * length * xcoord ** 2 + xcoord ** 3)
            / (24.0 * youngs_modulus * second_moment))


sample_points = np.linspace(0.0, length, 400)
fem_curve = basis.interpolator(x)(sample_points[None]).reshape(-1)
exact_curve = analytical_solution(sample_points)
midspan = length / 2.0
midspan_deflection = basis.interpolator(x)(np.array([[midspan]])).reshape(-1)[0]
midspan_exact = analytical_solution(np.array([midspan]))[0]
linf_error = np.max(np.abs(fem_curve - exact_curve))

print(f"midspan deflection (FEM): {midspan_deflection:.6e} m")
print(f"midspan deflection (exact): {midspan_exact:.6e} m")
print(f"maximum pointwise error: {linf_error:.6e} m")


def visualize():
    from skfem.visuals.matplotlib import plot

    ax = plot(basis, x, Nrefs=3, color='C0-')
    ax.plot(sample_points, exact_curve, 'k--', label='analytical')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('w(x) [m]')
    ax.set_title('Simply supported beam with uniform distributed load')
    ax.grid(alpha=0.3)
    ax.legend()

    return ax


if __name__ == '__main__':
    from os.path import splitext
    from sys import argv

    from matplotlib.pyplot import tight_layout
    from skfem.visuals.matplotlib import savefig

    visualize()
    tight_layout()
    savefig(f"{splitext(argv[0])[0]}_solution.png")