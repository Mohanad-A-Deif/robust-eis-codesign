"""2-D finite-element computation of the geometric conductance per unit gap length
for periodic coplanar electrodes (unit cell from the centre of one electrode to the
centre of the next), used to (i) verify the conformal-mapping cell constant and
(ii) build the higher-fidelity test generator (finite sample height)."""
import numpy as np
from skfem import MeshTri, Basis, ElementTriP2, BilinearForm, condense, solve, asm
from skfem.helpers import dot, grad
from functools import lru_cache


@BilinearForm
def laplace(u, v, w):
    return dot(grad(u), grad(v))


@lru_cache(maxsize=4096)
def flux_per_gap(w_um, s_um, H_um, nref=5):
    """Dimensionless conductance Phi (G' = sigma*Phi per unit gap length)."""
    w, s, H = w_um, s_um, H_um
    Lx = 0.5 * w + s + 0.5 * w
    # graded mesh: dense near y=0 and around electrode edges
    xs_edges = np.array([0.5 * w, 0.5 * w + s])
    x = np.unique(np.concatenate([np.linspace(0, Lx, 41),
                                  *[e + np.sign(np.linspace(-1, 1, 41)) * (np.linspace(-1, 1, 41) ** 2) * min(w, s) * 0.45 for e in xs_edges]]))
    x = x[(x >= 0) & (x <= Lx)]
    y = np.unique(np.concatenate([H * (np.linspace(0, 1, 40) ** 2.5), np.linspace(0, H, 12)]))
    m = MeshTri.init_tensor(x, y)
    m = m.refined(1) if nref > 0 else m
    basis = Basis(m, ElementTriP2())
    A = asm(laplace, basis)
    dofs = basis.get_dofs(lambda p: np.isclose(p[1], 0.0) & (p[0] <= 0.5 * w + 1e-12))
    dofsB = basis.get_dofs(lambda p: np.isclose(p[1], 0.0) & (p[0] >= 0.5 * w + s - 1e-12))
    u = np.zeros(basis.N)
    u[dofs.all()] = 1.0
    D = np.concatenate([dofs.all(), dofsB.all()])
    u = solve(*condense(A, x=u, D=D))
    return float(u @ (A @ u))   # = integral |grad u|^2 = conductance/sigma for unit voltage


if __name__ == "__main__":
    from scipy.special import ellipk
    def phi_analytic(w, s):
        k = np.cos(np.pi * w / (2 * (w + s))); m = k * k
        return ellipk(1 - m) / (2 * ellipk(m))
    for w, s in [(20, 20), (50, 50), (50, 150), (100, 30), (30, 100)]:
        for H in [5000, 100, 50]:
            print(w, s, H, round(flux_per_gap(w, s, H), 4), round(phi_analytic(w, s), 4))
