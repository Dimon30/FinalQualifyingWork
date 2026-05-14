"""
Утилиты визуализации результатов моделирования.
Стиль соответствует графикам в диссертации.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def ensure_out(outdir: str) -> None:
    """Создать директорию для результатов если не существует."""
    os.makedirs(outdir, exist_ok=True)


def display_path(p: str) -> str:
    """Вернуть путь для вывода в консоль относительно текущей рабочей директории.

    При запуске из корня проекта возвращает читаемый относительный путь,
    например ``code_app/out_images/ch4_line`` вместо абсолютного пути.
    """
    try:
        rel = os.path.relpath(p)
        return rel.replace("\\", "/")
    except ValueError:
        # На Windows os.path.relpath выбрасывает ValueError если пути на разных дисках.
        return p.replace("\\", "/")


def plot_3d_traj(
    p_ref: np.ndarray,
    p_real: np.ndarray,
    outpath: str,
    title: str = "",
) -> None:
    """3D график траектории (заданная пунктиром красным, реальная — синим)."""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    if p_ref is not None:
        ax.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2],
                "--r", linewidth=1.5, label="Заданная траектория")
    ax.plot(p_real[:, 0], p_real[:, 1], p_real[:, 2],
            color=(0.0078, 0.447, 0.741), linewidth=2.0,
            label="Траектория квадрокоптера")
    ax.set_xlabel("x, м"); ax.set_ylabel("y, м"); ax.set_zlabel("z, м")
    ax.legend(); ax.grid(True)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_errors(
    t: np.ndarray,
    e: np.ndarray,
    labels: list,
    outpath: str,
    title: str = "",
    ylim: tuple = None,
) -> None:
    """График ошибок регулирования."""
    fig, ax = plt.subplots(figsize=(10, 4))
    styles = ["--", "-.", "-", "-"]
    colors = [(0.466, 0.674, 0.188), (0.929, 0.694, 0.125), (0.0078, 0.447, 0.741), "r"]
    for i, lab in enumerate(labels):
        c = colors[i % len(colors)]
        s = styles[i % len(styles)]
        ax.plot(t, e[:, i], linewidth=2.0, label=lab, linestyle=s, color=c)
    ax.set_xlabel("t, с"); ax.grid(True, linestyle="--")
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_velocity(
    t: np.ndarray,
    vel: np.ndarray,
    Vstar: float,
    outpath: str,
    title: str = "Линейная скорость",
) -> None:
    """График линейной скорости с горизонтальной линией V*."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, vel, color=(0.0078, 0.447, 0.741), linewidth=2.0, label="v, м/с")
    ax.axhline(Vstar, color="r", linestyle="--", linewidth=1.5,
               label=f"V* = {Vstar}")
    ax.set_xlabel("t, с"); ax.grid(True, linestyle="--")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_angles(
    t: np.ndarray,
    angles: np.ndarray,
    outpath: str,
    title: str = "Угловые координаты",
) -> None:
    """График углов φ, θ, ψ."""
    fig, ax = plt.subplots(figsize=(8, 4))
    lbls = ["φ (рысканье)", "θ (тангаж)", "ψ (крен)"]
    styles = ["-", "--", "-."]
    for i in range(min(3, angles.shape[1])):
        ax.plot(t, angles[:, i], linewidth=2.0, label=lbls[i], linestyle=styles[i])
    ax.set_xlabel("t, с"); ax.set_ylabel("рад")
    ax.grid(True, linestyle="--"); ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_xy(
    p_ref: np.ndarray,
    p_real: np.ndarray,
    outpath: str,
    title: str = "Проекция X-Y",
) -> None:
    """Проекция траектории на плоскость XY."""
    fig, ax = plt.subplots(figsize=(6, 6))
    if p_ref is not None:
        ax.plot(p_ref[:, 0], p_ref[:, 1], "--r", linewidth=1.5,
                label="Заданная")
    ax.plot(p_real[:, 0], p_real[:, 1],
            color=(0.0078, 0.447, 0.741), linewidth=2.0,
            label="Квадрокоптер")
    ax.set_xlabel("x, м"); ax.set_ylabel("y, м")
    ax.grid(True, linestyle="--"); ax.legend()
    ax.set_title(title); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
