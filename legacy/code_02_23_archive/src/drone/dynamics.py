from __future__ import annotations

import numpy as np

from .utils import sat_tanh

G = 9.81


def thrust_direction(phi: float, theta: float, psi: float) -> np.ndarray:
    """Направление тяги b(φ,θ,ψ) как в модели в диссертации/Simulink.

    Важно по обозначениям:
      φ = yaw, θ = pitch, ψ = roll.

    b = [
        cφ sθ cψ + sφ sψ,
        sφ sθ cψ − cφ sψ,
        cθ cψ
    ]
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array(
        [cphi * sth * cps + sphi * sps,
         sphi * sth * cps - cphi * sps,
         cth * cps],
        dtype=float,
    )


def quad_dynamics_extended(x: np.ndarray, U: np.ndarray, L: float = 5.0) -> np.ndarray:
    """Расширенная модель (как в старом коде и в Simulink-примере главы 4).

    Состояние:
      x = [p(3), v(3), angles(3), rates(3), u1_bar, rho1, u2, rho2]
        p = (x,y,z) в ИСК (Earth frame)
        v = (vx,vy,vz)
        angles = (φ,θ,ψ) = (yaw,pitch,roll)
        rates = (φ̇,θ̇,ψ̇)
        u1 = sat_L(u1_bar) — виртуальное управление по тяге (в отклонениях)
        u1̇ = rho1, rho1̇ = v1
        u2̇ = rho2, rho2̇ = v2
      Управление:
        U = [v1, v2, u3, u4]
        φ̈ = u2, θ̈ = u3, ψ̈ = u4

    Линейное ускорение:
      a = b(φ,θ,ψ) * (u1 + g) - [0,0,g]
    """
    x = np.asarray(x, dtype=float).copy()
    U = np.asarray(U, dtype=float).copy()

    p = x[0:3]
    v = x[3:6]
    phi, theta, psi = x[6:9]
    phidot, thetadot, psidot = x[9:12]
    u1_bar, rho1, u2, rho2 = x[12], x[13], x[14], x[15]

    v1, v2, u3, u4 = U

    u1 = float(sat_tanh(u1_bar, L))
    b = thrust_direction(phi, theta, psi)
    a = b * (u1 + G) - np.array([0.0, 0.0, G], dtype=float)

    xdot = np.zeros_like(x)
    xdot[0:3] = v
    xdot[3:6] = a
    xdot[6] = phidot
    xdot[7] = thetadot
    xdot[8] = psidot
    xdot[9] = u2
    xdot[10] = u3
    xdot[11] = u4
    xdot[12] = rho1
    xdot[13] = v1
    xdot[14] = rho2
    xdot[15] = v2
    return xdot


def rk4_step(f, x: np.ndarray, u: np.ndarray, dt: float, **kwargs) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    k1 = f(x, u, **kwargs)
    k2 = f(x + 0.5 * dt * k1, u, **kwargs)
    k3 = f(x + 0.5 * dt * k2, u, **kwargs)
    k4 = f(x + dt * k3, u, **kwargs)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def prop_forces_from_virtual(u: np.ndarray, m: float, C: float, Jphi: float, Jtheta: float, Jpsi: float, ell: float) -> np.ndarray:
    """Преобразование (4.6): виртуальные управления u -> силы пропеллеров F.

    В диссертации:
      u = M * A * F - [g,0,0,0]^T
    где
      M = diag(1/m, C/Jphi, ell/Jtheta, ell/Jpsi)
      A = [[1,1,1,1],
           [1,-1,1,-1],
           [-1,1,1,-1],
           [-1,-1,1,1]]

    Отсюда F = A^{-1} * M^{-1} * (u + [g,0,0,0]).
    """
    u = np.asarray(u, dtype=float).reshape(4)
    M = np.diag([1.0/m, C/Jphi, ell/Jtheta, ell/Jpsi])
    A = np.array([[1, 1, 1, 1],
                  [1,-1,-1, 1],
                  [1, 1, 1, 1],  # placeholder, will overwrite below
                  [1,-1, 1,-1]], dtype=float)
    # Use exact matrix from dissertation snippet (4.6):
    A = np.array([[1,  1,  1,  1],
                  [1, -1,  1, -1],
                  [-1, 1,  1, -1],
                  [-1,-1,  1,  1]], dtype=float)

    b = u + np.array([G, 0.0, 0.0, 0.0], dtype=float)
    Fin = np.linalg.solve(A, np.linalg.solve(M, b))
    return Fin
