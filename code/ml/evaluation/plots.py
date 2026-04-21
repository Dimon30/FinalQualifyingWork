"""Визуализация результатов бенчмарка: time-series overlay и сводные bar charts.

Использование::

    from ml.evaluation.benchmark import ModelResult
    from ml.evaluation.plots import (
        plot_e2_comparison,
        plot_velocity_comparison,
        plot_summary_bar,
        save_latex_table,
    )
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.evaluation.benchmark import ModelResult


# ---------------------------------------------------------------------------
# Цвета и стили для каждой модели
# ---------------------------------------------------------------------------

_STYLE: dict[str, dict] = {
    "baseline": dict(color=(0.4, 0.4, 0.4),  ls="-",  lw=1.6, label="Константная $V^*$"),
    "mlp":      dict(color=(0.008, 0.447, 0.741), ls="-",  lw=1.8, label="MLP"),
    "sac":      dict(color=(0.85,  0.33,  0.10),  ls="--", lw=1.8, label="SAC"),
    "td3":      dict(color=(0.13,  0.63,  0.13),  ls="-.", lw=1.8, label="TD3"),
    "ppo":      dict(color=(0.58,  0.40,  0.74),  ls=":",  lw=2.0, label="PPO"),
}

_MODEL_ORDER = ["baseline", "mlp", "sac", "td3", "ppo"]


def _style(name: str) -> dict:
    return _STYLE.get(name, dict(color="black", ls="-", lw=1.5, label=name.upper()))


# ---------------------------------------------------------------------------
# Time-series: e2(t)
# ---------------------------------------------------------------------------

def plot_e2_comparison(
    results: list[ModelResult],
    scenario_name: str,
    out_dir: str,
    scenario_label: str = "",
) -> str:
    """e2(t) для всех моделей на одном сценарии.

    Возвращает путь к сохранённому файлу.
    """
    os.makedirs(out_dir, exist_ok=True)
    sc_results = [r for r in results if r.scenario_name == scenario_name]
    sc_results.sort(key=lambda r: _MODEL_ORDER.index(r.model_name)
                    if r.model_name in _MODEL_ORDER else 99)

    fig, ax = plt.subplots(figsize=(12, 4))
    for r in sc_results:
        s = _style(r.model_name)
        ax.plot(r.t, r.errors[:, 2],
                color=s["color"], ls=s["ls"], lw=s["lw"], label=s["label"])

    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("$t$, с")
    ax.set_ylabel("$e_2$, м")
    title = f"Поперечная ошибка $e_2$ — {scenario_label or scenario_name}"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, ls="--", alpha=0.45)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{scenario_name}_e2.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Time-series: velocity(t)
# ---------------------------------------------------------------------------

def plot_velocity_comparison(
    results: list[ModelResult],
    scenario_name: str,
    out_dir: str,
    scenario_label: str = "",
    Vstar_base: float = 1.0,
) -> str:
    """||v||(t) для всех моделей на одном сценарии."""
    os.makedirs(out_dir, exist_ok=True)
    sc_results = [r for r in results if r.scenario_name == scenario_name]
    sc_results.sort(key=lambda r: _MODEL_ORDER.index(r.model_name)
                    if r.model_name in _MODEL_ORDER else 99)

    fig, ax = plt.subplots(figsize=(12, 4))
    for r in sc_results:
        s = _style(r.model_name)
        ax.plot(r.t, r.velocity,
                color=s["color"], ls=s["ls"], lw=s["lw"], label=s["label"])

    ax.set_xlabel("$t$, с")
    ax.set_ylabel("$\\|v\\|$, м/с")
    title = f"Линейная скорость — {scenario_label or scenario_name}"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, ls="--", alpha=0.45)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{scenario_name}_velocity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Grouped bar chart — сводная метрика по всем сценариям
# ---------------------------------------------------------------------------

def plot_summary_bar(
    results: list[ModelResult],
    metric: str,
    out_dir: str,
    ylabel: str = "",
    title: str = "",
    log_scale: bool = False,
) -> str:
    """Grouped bar chart: metric по (сценарий × модель).

    Параметры:
        metric — имя атрибута ModelResult ('e2_rms', 'v_mean', 'speedup', …)
        ylabel — подпись оси Y
        title  — заголовок графика
        log_scale — логарифмическая ось Y
    """
    os.makedirs(out_dir, exist_ok=True)

    # Собираем уникальные сценарии и модели (кроме baseline для speedup)
    scenarios = list(dict.fromkeys(r.scenario_name for r in results))
    models = [m for m in _MODEL_ORDER if m != "baseline" and
              any(r.model_name == m for r in results)]
    if not models:
        models = list(dict.fromkeys(
            r.model_name for r in results if r.model_name != "baseline"
        ))

    # Добавить baseline если нужно
    if metric != "speedup":
        models = ["baseline"] + [m for m in models if m != "baseline"]

    n_sc = len(scenarios)
    n_m  = len(models)
    x    = np.arange(n_sc)
    width = 0.8 / n_m

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_sc), 5))

    for i, model in enumerate(models):
        vals = []
        for sc in scenarios:
            match = [r for r in results
                     if r.scenario_name == sc and r.model_name == model]
            vals.append(getattr(match[0], metric) if match else 0.0)

        s = _style(model)
        bars = ax.bar(
            x + i * width - (n_m - 1) * width / 2,
            vals, width * 0.92,
            color=s["color"], label=s["label"], alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"Сравнение: {metric}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    if log_scale:
        ax.set_yscale("log")
    fig.tight_layout()

    path = os.path.join(out_dir, f"summary_{metric}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Автогенерация LaTeX-таблицы
# ---------------------------------------------------------------------------

_MODEL_LABEL = {
    "baseline": "Константная $V^*$",
    "mlp":      "MLP",
    "sac":      "SAC",
    "td3":      "TD3",
    "ppo":      "PPO",
}

_SCENARIO_LABEL = {
    "spiral_r3":   "Спираль $r{=}3$",
    "circle_r3z5": "Окружность $r{=}3$",
    "helix_r2":    "Спираль $r{=}2$",
    "line_diag":   "Прямая $x{=}s,y{=}s,z{=}s$",
}


def save_latex_table(
    results: list[ModelResult],
    path: str,
) -> None:
    """Сгенерировать LaTeX-таблицу метрик и сохранить в файл.

    Структура таблицы:
    Кривая | Модель | e1_rms | e2_rms | e2_max | v_mean | ускорение | сошлось

    Использует longtable + multirow.
    """
    scenarios = list(dict.fromkeys(r.scenario_name for r in results))
    models    = [m for m in _MODEL_ORDER
                 if any(r.model_name == m for r in results)]

    lines = [
        r"\begin{longtable}{llrrrrrr}",
        r"\caption{Сравнительные показатели моделей оптимизации $V^*$ на тестовом наборе кривых}"
        r"\label{tab:model_benchmark}\\",
        r"\toprule",
        r"\multirow{2}{*}{Кривая} & \multirow{2}{*}{Модель} & "
        r"$e_{1,\mathrm{RMS}}$, & $e_{2,\mathrm{RMS}}$, & $e_{2,\max}$, & "
        r"$\bar{v}$, & Ускорение & Сх- \\",
        r"& & м & м & м & м/с & $\times$ & лось \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{c}{\tablename\ \thetable{} — продолжение} \\",
        r"\toprule",
        r"\multirow{2}{*}{Кривая} & \multirow{2}{*}{Модель} & "
        r"$e_{1,\mathrm{RMS}}$, & $e_{2,\mathrm{RMS}}$, & $e_{2,\max}$, & "
        r"$\bar{v}$, & Ускорение & Сх- \\",
        r"& & м & м & м & м/с & $\times$ & лось \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{8}{r}{\textit{Продолжение на следующей странице}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for sc_idx, sc in enumerate(scenarios):
        sc_label = _SCENARIO_LABEL.get(sc, sc)
        sc_results_all = [r for r in results if r.scenario_name == sc]
        sc_models = [m for m in models
                     if any(r.model_name == m for r in sc_results_all)]
        n_rows = len(sc_models)

        for m_idx, model in enumerate(sc_models):
            match = [r for r in sc_results_all if r.model_name == model]
            if not match:
                continue
            r = match[0]
            m_label = _MODEL_LABEL.get(model, model.upper())

            conv_str = r"\checkmark" if r.converged else r"$\times$"
            sp_str   = f"{r.speedup:.2f}" if model != "baseline" else "---"

            if m_idx == 0:
                curve_cell = f"\\multirow{{{n_rows}}}{{*}}{{{sc_label}}}"
            else:
                curve_cell = ""

            row = (
                f"  {curve_cell} & {m_label} "
                f"& {r.e1_rms:.4f} & {r.e2_rms:.4f} & {r.e2_max:.4f} "
                f"& {r.v_mean:.3f} & {sp_str} & {conv_str} \\\\"
            )
            lines.append(row)

        if sc_idx < len(scenarios) - 1:
            lines.append(r"  \midrule")

    lines.append(r"\end{longtable}")

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
