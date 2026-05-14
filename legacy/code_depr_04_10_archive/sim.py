from __future__ import annotations
import numpy as np
from typing import Callable, Dict

from integrators import rk4_step
from dynamics import quad_dynamics_extended

def simulate(controller_step: Callable[[float, np.ndarray, np.ndarray, float], np.ndarray], x0: np.ndarray, T: float, dt: float, L: float = 5.0) -> Dict[str, np.ndarray]:
    n = int(T/dt)+1
    t = np.linspace(0.0, T, n)
    x = np.zeros((n, len(x0)))
    U = np.zeros((n, 4))
    x[0] = x0
    for k in range(n-1):
        U[k] = controller_step(t[k], x[k], U[k-1] if k>0 else np.zeros(4), dt)
        x[k+1] = rk4_step(lambda xx, uu, **kw: quad_dynamics_extended(xx, uu, L=L), t[k], x[k], U[k], dt)
    U[-1] = U[-2]
    return {"t": t, "x": x, "U": U}
