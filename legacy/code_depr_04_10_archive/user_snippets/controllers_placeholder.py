from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from trajectories import TrajPoint


@dataclass(frozen=True)
class PDGains:
    kp_pos: float = 6.0
    kd_pos: float = 4.0
    kp_ang: float = 10.0
    kd_ang: float = 6.0


def simple_pd_controller(
    traj: Callable[[float], TrajPoint],
    gains: PDGains = PDGains(),
    yaw_ref: float = 0.0,
) -> Callable[[float, np.ndarray], Tuple[float, float, float, float]]:
    """Простой трекинг-контроллер (как заглушка), чтобы оживить модель в Python.

    ВАЖНО: это НЕ тот регулятор из диссертации (глава 2–4). Он нужен, чтобы:
    - проверить корректность портирования динамики;
    - получить траектории и графики, похожие по смыслу на Simulink.

    Идея:
      a_cmd = a_ref + Kp*(p_ref - p) + Kd*(v_ref - v)
      thrust_vec = a_cmd + [0,0,g]
      (u1+g) = ||thrust_vec||, b = thrust_vec/||thrust_vec||
      дальше подбираем phi_ref, theta_ref (при psi_ref=yaw_ref) и стабилизируем углы
      в каналах u2,u3,u4 как угловые ускорения.
    """
    def u(t: float, x: np.ndarray) -> Tuple[float, float, float, float]:
        p = x[0:3]
        v = x[3:6]
        phi, theta, psi = x[6:9]
        phidot, thetadot, psidot = x[9:12]

        tp = traj(t)
        e_p = tp.p - p
        e_v = tp.v - v

        a_cmd = tp.a + gains.kp_pos * e_p + gains.kd_pos * e_v

        g_vec = np.array([0.0, 0.0, 9.81], dtype=float)
        thrust_vec = a_cmd + g_vec
        norm = float(np.linalg.norm(thrust_vec) + 1e-9)
        b = thrust_vec / norm

        # Выбор желаемых углов. Для простоты считаем psi ~= yaw_ref (часто так и задают).
        # Инверсия b(phi,theta,psi) в общем виде громоздка; в демо берём yaw_ref=0.
        # Тогда b = [cphi*sth, sphi*sth, cth].
        psi_ref = yaw_ref
        if abs(psi_ref) > 1e-6:
            # Чтобы не вводить сложную геометрию, просто зафиксируем psi_ref=0,
            # а yaw-канал стабилизируем к yaw_ref отдельным ПД.
            psi_ref_eff = 0.0
        else:
            psi_ref_eff = 0.0

        bx, by, bz = b
        theta_ref = float(np.arctan2(np.sqrt(bx*bx + by*by), bz))
        phi_ref = float(np.arctan2(by, bx))

        # Угловые каналы: задаём желаемые угловые ускорения
        u2 = gains.kp_ang * (phi_ref - phi) + gains.kd_ang * (0.0 - phidot)
        u3 = gains.kp_ang * (theta_ref - theta) + gains.kd_ang * (0.0 - thetadot)
        u4 = gains.kp_ang * (yaw_ref - psi) + gains.kd_ang * (0.0 - psidot)

        u1 = norm - 9.81
        return float(u1), float(u2), float(u3), float(u4)

    return u
