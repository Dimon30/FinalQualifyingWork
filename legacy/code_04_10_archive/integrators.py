from __future__ import annotations
import numpy as np

def rk4_step(f, t: float, x: np.ndarray, u: np.ndarray, dt: float, **kwargs) -> np.ndarray:
    k1 = f(x, u, **kwargs)
    k2 = f(x + 0.5*dt*k1, u, **kwargs)
    k3 = f(x + 0.5*dt*k2, u, **kwargs)
    k4 = f(x + dt*k3, u, **kwargs)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
