from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class HighGainParams:
    kappa: float = 100.0
    a: tuple = (5.0,10.0,10.0,5.0,1.0)
    gamma: tuple = (1.0,4.0,6.0,4.0,1.0)
    L: float = 5.0

def sat_vec_tanh(x: np.ndarray, L: float) -> np.ndarray:
    return L * np.tanh(x / max(L, 1e-9))

class DerivativeObserver4:
    def __init__(self, dim: int, p: HighGainParams):
        self.dim = dim
        self.p = p
        self.x1 = np.zeros(dim)
        self.x2 = np.zeros(dim)
        self.x3 = np.zeros(dim)
        self.x4 = np.zeros(dim)
        self.sigma = np.zeros(dim)

    def step(self, y: np.ndarray, y4_model: np.ndarray, dt: float) -> None:
        k = self.p.kappa
        a1,a2,a3,a4,a5 = self.p.a
        e = (y - self.x1)
        self.x1 += dt*(self.x2 + k*a1*e)
        self.x2 += dt*(self.x3 + (k**2)*a2*e)
        self.x3 += dt*(self.x4 + (k**3)*a3*e)
        self.x4 += dt*(self.sigma + y4_model + (k**4)*a4*e)
        self.sigma += dt*((k**5)*a5*e)

    def hat(self):
        return self.x1, self.x2, self.x3, self.x4, self.sigma
