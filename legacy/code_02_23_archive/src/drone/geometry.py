from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """Гладкая пространственная кривая P_s = η(s)."""
    p: Callable[[float], np.ndarray]       # position η(s)
    dp: Callable[[float], np.ndarray]      # first derivative dη/ds
    ddp: Callable[[float], np.ndarray]     # second derivative d²η/ds² (для Ньютона)

    def tangent(self, s: float) -> np.ndarray:
        t = np.asarray(self.dp(s), dtype=float)
        n = np.linalg.norm(t)
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return t / n

    def yaw_star(self, s: float) -> float:
        """φ*(s) — курс вдоль проекции касательной на OXY."""
        t = self.tangent(s)
        return float(np.arctan2(t[1], t[0]))

    def beta(self, s: float) -> float:
        """β(s) — угол наклона касательной относительно плоскости OXY."""
        t = self.tangent(s)
        return float(np.arctan2(t[2], np.sqrt(t[0] ** 2 + t[1] ** 2)))

    def eps(self, s: float, h: float = 1e-4) -> float:
        """ε(s) ≈ dφ*/ds (численно)."""
        a1 = self.yaw_star(s - h)
        a2 = self.yaw_star(s + h)
        da = float(np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1)))
        return da / (2 * h)


def Rz(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0.0],
                     [sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def Ry(b: float) -> np.ndarray:
    cb, sb = np.cos(b), np.sin(b)
    return np.array([[cb, 0.0, sb],
                     [0.0, 1.0, 0.0],
                     [-sb, 0.0, cb]], dtype=float)


def se_from_pose(p_xyz: np.ndarray, s: float, traj: Trajectory) -> Tuple[float, float, float]:
    """(s_local, e1, e2) по формуле (60): поворот Rz(α) затем Ry(β)."""
    ps = np.asarray(traj.p(s), dtype=float).reshape(3)
    alpha = traj.yaw_star(s)
    beta = traj.beta(s)
    v = np.asarray(p_xyz, dtype=float).reshape(3) - ps
    q = Ry(beta).T @ (Rz(alpha).T @ v)
    return float(q[0]), float(q[1]), float(q[2])


def nearest_s_newton(
    p_xyz: np.ndarray,
    traj: Trajectory,
    s0: float,
    iters: int = 10,
    damping: float = 0.7,
) -> float:
    """Численный поиск ближайшей точки на кривой: минимизация ||η(s)-P||^2.

    Решаем f(s)=d/ds ||η(s)-P||^2 = 2 (η(s)-P)·η'(s) = 0 методом Ньютона.
    """
    s = float(s0)
    P = np.asarray(p_xyz, dtype=float).reshape(3)
    for _ in range(iters):
        r = np.asarray(traj.p(s), dtype=float).reshape(3) - P
        dp = np.asarray(traj.dp(s), dtype=float).reshape(3)
        ddp = np.asarray(traj.ddp(s), dtype=float).reshape(3)
        f = 2.0 * float(r @ dp)
        fp = 2.0 * float(dp @ dp + r @ ddp)
        if abs(fp) < 1e-9:
            break
        step = f / fp
        s -= damping * step
        if not np.isfinite(s):
            s = float(s0)
            break
    return float(s)



def line_traj() -> Trajectory:
    # направление (1,1,1) нормировано => ds = dl
    k = 1.0 / np.sqrt(3.0)
    def p(s): return np.array([k*s, k*s, k*s], dtype=float)
    def dp(s): return np.array([k, k, k], dtype=float)
    def ddp(s): return np.zeros(3, dtype=float)
    return Trajectory(p=p, dp=dp, ddp=ddp)


def helix_traj(r: float = 3.0) -> Trajectory:
    # исходная спираль: [r cos u, r sin u, u], ||d/du|| = sqrt(r^2+1)=c.
    # берём s как длину дуги: u = s/c
    c = float(np.sqrt(r*r + 1.0))
    def p(s):
        u = s / c
        return np.array([r*np.cos(u), r*np.sin(u), u], dtype=float)
    def dp(s):
        u = s / c
        du = 1.0 / c
        return np.array([-r*np.sin(u)*du, r*np.cos(u)*du, du], dtype=float)
    def ddp(s):
        u = s / c
        du = 1.0 / c
        return np.array([-r*np.cos(u)*(du**2), -r*np.sin(u)*(du**2), 0.0], dtype=float)
    return Trajectory(p=p, dp=dp, ddp=ddp)
