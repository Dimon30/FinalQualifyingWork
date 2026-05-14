from __future__ import annotations
import numpy as np

G = 9.81

def thrust_direction(phi: float, theta: float, psi: float) -> np.ndarray:
    """Направление тяги b(φ,θ,ψ) из диссертации (см. (4.22)).

    ВАЖНО: φ=yaw, θ=pitch, ψ=roll.

    b = [ cφ sθ cψ + sφ sψ,
          sφ sθ cψ − cφ sψ,
          cθ cψ ]
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array([
        cphi*sth*cps + sphi*sps,
        sphi*sth*cps - cphi*sps,
        cth*cps
    ], dtype=float)

def sat_tanh(x: float, L: float) -> float:
    return float(L * np.tanh(x / max(L, 1e-9)))

def quad_dynamics_extended(x: np.ndarray, U: np.ndarray, L: float = 5.0) -> np.ndarray:
    """Расширенная модель для глав 3–4.

    x = [p(3), v(3), angles(3), rates(3), u1_bar, rho1, u2, rho2]
    U = [v1, v2, u3, u4]

    u1 = sat_L(u1_bar), u1dot=rho1, rho1dot=v1; u2dot=rho2, rho2dot=v2.
    """
    p = x[0:3]
    v = x[3:6]
    phi, theta, psi = x[6:9]
    phidot, thetadot, psidot = x[9:12]
    u1_bar, rho1, u2, rho2 = x[12], x[13], x[14], x[15]

    v1, v2, u3, u4 = U

    u1 = sat_tanh(u1_bar, L)
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
