"""
Универсальный симулятор для всех глав.

Функция simulate() принимает функцию шага регулятора и выполняет
численное интегрирование методом RK4.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict

from integrators import rk4_step


def simulate(
    dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    controller_step: Callable[[float, np.ndarray, np.ndarray, float], np.ndarray],
    x0: np.ndarray,
    T: float,
    dt: float,
) -> Dict[str, np.ndarray]:
    """Симуляция замкнутой системы.

    Аргументы:
        dynamics_fn      — правая часть ОДУ f(x, U) → ẋ
        controller_step  — шаг регулятора: (t, x, U_prev, dt) → U
        x0               — начальное состояние
        T                — время симуляции (с)
        dt               — шаг интегрирования (с)

    Возвращает словарь:
        't' — массив времени [n]
        'x' — траектория состояния [n × dim]
        'U' — управляющие воздействия [n × udim]
    """
    n = int(T / dt) + 1
    t_arr = np.linspace(0.0, T, n)
    x_arr = np.zeros((n, len(x0)))
    x_arr[0] = x0

    # Определяем размерность управления по первому шагу
    U0 = controller_step(t_arr[0], x0, None, dt)
    U_arr = np.zeros((n, len(U0)))
    U_arr[0] = U0

    for k in range(n - 1):
        U_arr[k] = controller_step(t_arr[k], x_arr[k], U_arr[k-1] if k > 0 else None, dt)
        x_arr[k+1] = rk4_step(dynamics_fn, t_arr[k], x_arr[k], U_arr[k], dt)

    U_arr[-1] = U_arr[-2]
    return {"t": t_arr, "x": x_arr, "U": U_arr}
