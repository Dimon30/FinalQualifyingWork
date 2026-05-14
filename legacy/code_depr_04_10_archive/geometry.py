from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable

@dataclass(frozen=True)
class CurveGeom:
    p: Callable[[float], np.ndarray]
    t: Callable[[float], np.ndarray]
    yaw_star: Callable[[float], float]
    beta: Callable[[float], float]
    eps: Callable[[float], float]

def Rz(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca,-sa,0.0],[sa,ca,0.0],[0.0,0.0,1.0]], dtype=float)

def Ry(b: float) -> np.ndarray:
    cb, sb = np.cos(b), np.sin(b)
    return np.array([[cb,0.0,sb],[0.0,1.0,0.0],[-sb,0.0,cb]], dtype=float)

def se_from_pose(p_xyz: np.ndarray, s: float, curve: CurveGeom) -> Tuple[float,float,float]:
    ps = curve.p(s)
    alpha = curve.yaw_star(s)
    beta = curve.beta(s)
    v = p_xyz - ps
    q = Ry(beta).T @ (Rz(alpha).T @ v)
    return float(q[0]), float(q[1]), float(q[2])

def line_xyz_curve() -> CurveGeom:
    def p(s): return np.array([s,s,s], dtype=float)
    def t(s): return np.array([1.0,1.0,1.0], dtype=float)
    def yaw_star(s): return 0.0
    def beta(s):
        tt = t(s)
        return float(np.arctan2(tt[2], np.sqrt(tt[0]**2 + tt[1]**2)))
    def eps(s): return 0.0
    return CurveGeom(p=p, t=t, yaw_star=yaw_star, beta=beta, eps=eps)

def spiral_curve(r: float = 3.0) -> CurveGeom:
    def p(s): return np.array([r*np.cos(s), r*np.sin(s), s], dtype=float)
    def t(s): return np.array([-r*np.sin(s), r*np.cos(s), 1.0], dtype=float)
    def yaw_star(s):
        tt = t(s)
        return float(np.arctan2(tt[1], tt[0]))
    def beta(s):
        tt = t(s)
        return float(np.arctan2(tt[2], np.sqrt(tt[0]**2 + tt[1]**2)))
    def eps(s):
        h = 1e-5
        a1 = yaw_star(s-h)
        a2 = yaw_star(s+h)
        da = np.arctan2(np.sin(a2-a1), np.cos(a2-a1))
        return float(da/(2*h))
    return CurveGeom(p=p, t=t, yaw_star=yaw_star, beta=beta, eps=eps)

def nearest_point_line(p_xyz: np.ndarray) -> float:
    x,y,z = map(float, p_xyz)
    return (x+y+z)/3.0

def spiral_nearest_observer_step(zeta: float, p_xyz: np.ndarray, r: float = 3.0, gamma: float = 1.0, dt: float = 0.01) -> float:
    x,y,z = map(float, p_xyz)
    H = r*np.sin(zeta)*x - r*np.cos(zeta)*y - z + zeta
    rho = np.sign(1.0 + r*np.cos(zeta)*x + r*np.sin(zeta)*y)
    return float(zeta + (-gamma * rho * H) * dt)
