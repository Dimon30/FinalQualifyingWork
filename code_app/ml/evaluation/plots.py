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
    os.makedirs(out_dir, exist_ok=True)

    scenarios = list(dict.fromkeys(r.scenario_name for r in results))
    models = [m for m in _MODEL_ORDER if m != "baseline" and
              any(r.model_name == m for r in results)]

    if not models:
        models = list(dict.fromkeys(
            r.model_name for r in results if r.model_name != "baseline"
        ))

    if metric != "speedup":
        models = ["baseline"] + [m for m in models if m != "baseline"]

    n_sc = len(scenarios)
    n_m = len(models)
    x = np.arange(n_sc)
    width = 0.8 / n_m

    y_clip = 0.6 if metric == "e2_rms" else None

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_sc), 5))

    for i, model in enumerate(models):
        vals = []
        for sc in scenarios:
            match = [r for r in results
                     if r.scenario_name == sc and r.model_name == model]
            vals.append(getattr(match[0], metric) if match else 0.0)

        plot_vals = [
            min(v, y_clip) if y_clip is not None else v
            for v in vals
        ]

        s = _style(model)

        bars = ax.bar(
            x + i * width - (n_m - 1) * width / 2,
            plot_vals,
            width * 0.92,
            color=s["color"],
            label=s["label"],
            alpha=0.85,
        )

        for bar, real_val, plot_val in zip(bars, vals, plot_vals):
            x_pos = bar.get_x() + bar.get_width() / 2

            if y_clip is not None and real_val > y_clip:
                ax.text(
                    x_pos,
                    y_clip + 0.025,
                    f"{real_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="red",
                    rotation=90,
                )
            else:
                ax.text(
                    x_pos,
                    plot_val + 0.01,
                    f"{real_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"Сравнение: {metric}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", ls="--", alpha=0.4)

    if metric == "e2_rms":
        ax.set_ylim(0, 0.75)
        ax.text(
            0.01,
            0.96,
            "Высокие значения обрезаны по оси Y, подписи показывают реальные значения",
            transform=ax.transAxes,
            fontsize=8,
            color="red",
            va="top",
        )

    if log_scale:
        ax.set_yscale("log")

    fig.tight_layout()

    path = os.path.join(out_dir, f"summary_{metric}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Автогенерация LaTeX-таблицы
# ---------------------------------------------------------------------------

_MODEL_LABEL = {
    "baseline": "Константное $V^*$",
    "mlp": "MLP",
    "sac": "SAC",
    "td3": "TD3",
    "ppo": "PPO",
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
    scenarios = list(dict.fromkeys(r.scenario_name for r in results))
    models = [m for m in _MODEL_ORDER if any(r.model_name == m for r in results)]

    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{longtable}{|p{2.45cm}|p{2.3cm}|r|r|r|r|r|c|}",
        r"\caption{Сравнительные показатели моделей оптимизации $V^*$ на тестовом наборе кривых}"
        r"\label{tab:model_benchmark}\\",
        r"\hline",
        r"\textbf{Кривая} & \textbf{Модель} & "
        r"\textbf{$e_{1,\mathrm{RMS}}$, м} & "
        r"\textbf{$e_{2,\mathrm{RMS}}$, м} & "
        r"\textbf{$e_{2,\max}$, м} & "
        r"\textbf{$\bar{v}$, м/с} & "
        r"\textbf{Ускорение, м/$\text{с}^2$} & "
        r"\textbf{Сошлось} \\",
        r"\hline",
        r"\endfirsthead",

        r"\multicolumn{8}{|c|}{\tablename\ \thetable{} — продолжение} \\",
        r"\hline",
        r"\textbf{Кривая} & \textbf{Модель} & "
        r"\textbf{$e_{1,\mathrm{RMS}}$, м} & "
        r"\textbf{$e_{2,\mathrm{RMS}}$, м} & "
        r"\textbf{$e_{2,\max}$, м} & "
        r"\textbf{$\bar{v}$, м/с} & "
        r"\textbf{Ускорение, м/$\text{с}^2$} & "
        r"\textbf{Сошлось} \\",
        r"\hline",
        r"\endhead",

        r"\hline",
        r"\multicolumn{8}{|r|}{\textit{Продолжение на следующей странице}} \\",
        r"\endfoot",

        r"\hline",
        r"\endlastfoot",
    ]

    for sc_idx, sc in enumerate(scenarios):
        sc_label = _SCENARIO_LABEL.get(sc, sc)
        sc_results_all = [r for r in results if r.scenario_name == sc]
        sc_models = [m for m in models if any(r.model_name == m for r in sc_results_all)]

        for m_idx, model in enumerate(sc_models):
            match = [r for r in sc_results_all if r.model_name == model]
            if not match:
                continue

            r = match[0]

            if model == "baseline":
                m_label = r"$V^*{=}\mathrm{const}$"
            else:
                m_label = _MODEL_LABEL.get(model, model.upper())

            conv_str = r"\checkmark" if r.converged else r"$\times$"
            sp_str = f"{r.speedup:.2f}" if model != "baseline" else "---"

            curve_cell = sc_label if m_idx == 0 else ""

            row = (
                f"  {curve_cell} & {m_label} "
                f"& {r.e1_rms:.3f} "
                f"& {r.e2_rms:.3f} "
                f"& {r.e2_max:.3f} "
                f"& {r.v_mean:.2f} "
                f"& {sp_str} "
                f"& {conv_str} \\\\"
            )
            lines.append(row)

        if sc_idx < len(scenarios) - 1:
            lines.append(r"  \hline")

    lines.extend([
        r"\end{longtable}",
        r"\endgroup",
    ])

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")