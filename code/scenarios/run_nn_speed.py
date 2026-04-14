"""
Симуляция дрона с нейросетевым оптимизатором скорости V*.

Режимы:
    --model path/to/model.pt   — использовать конкретную модель
    --model auto               — найти последнюю модель в code/ml/data/model/
    --no-nn                    — запуск только без NN (baseline)
    --compare                  — запустить оба варианта и сравнить (по умолчанию)

Кривая:
    --curve spiral             — спираль r=3 (из диссертации, по умолчанию)
    --curve line               — прямая x=s,y=s,z=s
    --curve circle             — окружность r=3

Запуск (из корня проекта):
    python code/scenarios/run_nn_speed.py
    python code/scenarios/run_nn_speed.py --curve spiral --model auto --compare
    python code/scenarios/run_nn_speed.py --model code/ml/data/model/speed_model.pt
    python code/scenarios/run_nn_speed.py --no-nn
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from drone_sim import make_curve, SimConfig, simulate_path_following
from drone_sim.geometry.curves import (
    spiral_curve, line_xyz_curve, nearest_point_line, CurveGeom,
)
from drone_sim.models.quad_model import QuadModel
from drone_sim.simulation.path_sim import SimResult
from drone_sim.visualization.plotting import ensure_out

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_DEFAULT_MODEL = os.path.join(_HERE, "..", "ml", "data", "model", "speed_model.pt")
_DEFAULT_OUT   = os.path.join(_HERE, "..", "out_images", "nn_speed")


# ---------------------------------------------------------------------------
# Поиск последней модели
# ---------------------------------------------------------------------------

def _find_latest_model(search_dir: str) -> str | None:
    """Найти самый свежий .pt файл в директории."""
    candidates = []
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.endswith(".pt"):
                p = os.path.join(root, f)
                candidates.append((os.path.getmtime(p), p))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def resolve_model_path(arg: str) -> str | None:
    """Разрешить путь к модели: 'auto' → поиск, иначе — прямой путь."""
    if arg == "auto":
        search = os.path.join(_HERE, "..", "ml", "data")
        found = _find_latest_model(search)
        if found is None:
            print("  [WARN] Модели не найдены в code/ml/data/. Запуск без NN.")
        else:
            print(f"  Авто-выбор модели: {found}")
        return found
    if os.path.isfile(arg):
        return arg
    print(f"  [WARN] Файл модели не найден: {arg}. Запуск без NN.")
    return None


# ---------------------------------------------------------------------------
# Конфигурация кривых
# ---------------------------------------------------------------------------

def make_scenario_curve(name: str) -> tuple[CurveGeom, dict]:
    """Вернуть (curve, cfg_kwargs) для заданного имени кривой."""
    if name == "spiral":
        curve = spiral_curve(r=3.0)
        x0 = np.zeros(16); x0[0:3] = [2.9, 0.0, 0.0]
        cfg_kw = dict(
            T=40.0, dt=0.002, kappa=200.0,
            gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0,
            x0=x0,
        )
        label = "Спираль r=3"
    elif name == "line":
        curve = line_xyz_curve()
        x0 = np.zeros(16); x0[0:3] = [1.0, 1.0, 0.0]
        cfg_kw = dict(
            T=30.0, dt=0.005, kappa=100.0,
            gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0,
            nearest_fn=nearest_point_line,
            x0=x0,
        )
        label = "Прямая x=s,y=s,z=s"
    elif name == "circle":
        curve = make_curve(lambda s: np.array([3.0 * np.cos(s / 3.0),
                                                3.0 * np.sin(s / 3.0), 0.0]))
        x0 = np.zeros(16); x0[0:3] = [2.9, 0.0, 0.0]
        cfg_kw = dict(
            T=40.0, dt=0.002, kappa=200.0,
            gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0,
            x0=x0,
        )
        label = "Окружность r=3"
    else:
        raise ValueError(f"Неизвестная кривая: {name!r}. Допустимо: spiral, line, circle")

    return curve, cfg_kw, label


# ---------------------------------------------------------------------------
# Запуск симуляции
# ---------------------------------------------------------------------------

def run_simulation(
    curve: CurveGeom,
    cfg_kw: dict,
    Vstar: float,
    drone: QuadModel,
    speed_fn=None,
    label: str = "",
) -> SimResult:
    """Запустить одну симуляцию. Возвращает SimResult."""
    cfg = SimConfig(
        Vstar=Vstar,
        quad_model=drone,
        speed_fn=speed_fn,
        **cfg_kw,
    )
    print(f"  Симуляция [{label}]: T={cfg.T}s  dt={cfg.dt}  kappa={cfg.kappa}", end="", flush=True)
    result = simulate_path_following(curve, cfg)
    e2f = result.errors[-1, 2]
    vf = result.velocity[-1]
    print(f"  → e2_final={e2f:+.4f}  vel_final={vf:.3f}")
    return result


# ---------------------------------------------------------------------------
# Визуализация сравнения
# ---------------------------------------------------------------------------

def plot_comparison(
    t_base: np.ndarray,
    r_base: SimResult,
    t_nn: np.ndarray | None,
    r_nn: SimResult | None,
    Vstar_base: float,
    out_dir: str,
    curve_label: str,
) -> None:
    """Сравнительные графики: baseline (константная V*) vs NN."""
    has_nn = r_nn is not None
    os.makedirs(out_dir, exist_ok=True)

    colors = {
        "base": (0.0078, 0.447, 0.741),   # синий
        "nn":   (0.85, 0.33, 0.10),        # оранжевый
        "ref":  (0.5, 0.5, 0.5),
    }

    def _label(tag, suffix=""):
        return f"{'Константная V*' if tag=='base' else 'NN V*'}{suffix}"

    # -----------------------------------------------------------------------
    # 1. Ошибки e2 (боковая) и e1 (тангенциальная)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # e1
    ax = axes[0]
    ax.plot(t_base, r_base.errors[:, 1], color=colors["base"],
            linewidth=1.8, label=_label("base"))
    if has_nn:
        ax.plot(t_nn, r_nn.errors[:, 1], color=colors["nn"],
                linewidth=1.8, linestyle="--", label=_label("nn"))
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_ylabel("e1, м")
    ax.set_title(f"Тангенциальная ошибка e1 — {curve_label}")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)

    # e2
    ax = axes[1]
    ax.plot(t_base, r_base.errors[:, 2], color=colors["base"],
            linewidth=1.8, label=_label("base"))
    if has_nn:
        ax.plot(t_nn, r_nn.errors[:, 2], color=colors["nn"],
                linewidth=1.8, linestyle="--", label=_label("nn"))
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("t, с"); ax.set_ylabel("e2, м")
    ax.set_title("Поперечная ошибка e2")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    p = os.path.join(out_dir, "compare_errors.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Ошибки: {p}")

    # -----------------------------------------------------------------------
    # 2. Скорости
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_base, r_base.velocity, color=colors["base"],
            linewidth=1.8, label=_label("base", f" (V*={Vstar_base})"))
    if has_nn:
        ax.plot(t_nn, r_nn.velocity, color=colors["nn"],
                linewidth=1.8, linestyle="--", label=_label("nn", " (адаптивная)"))
    ax.axhline(Vstar_base, color=colors["ref"], linestyle=":",
               linewidth=1.5, label=f"Baseline V*={Vstar_base}")
    ax.set_xlabel("t, с"); ax.set_ylabel("||v||, м/с")
    ax.set_title(f"Линейная скорость — {curve_label}")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(out_dir, "compare_velocity.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Скорости: {p}")

    # -----------------------------------------------------------------------
    # 3. 3D траектории
    # -----------------------------------------------------------------------
    n_cols = 2 if has_nn else 1
    fig = plt.figure(figsize=(7 * n_cols, 6))

    def _plot_3d(ax3d, result: SimResult, title: str, color):
        zeta_arr = result.zeta
        p_ref = np.stack([result.curve.p(z) for z in zeta_arr], axis=0)
        ax3d.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2],
                  "--", color=colors["ref"], linewidth=1.2, label="Заданная")
        ax3d.plot(result.x[:, 0], result.x[:, 1], result.x[:, 2],
                  color=color, linewidth=1.8, label="Дрон")
        ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
        ax3d.set_title(title)
        ax3d.legend(fontsize=8)

    ax1 = fig.add_subplot(1, n_cols, 1, projection="3d")
    _plot_3d(ax1, r_base, f"Константная V*={Vstar_base}", colors["base"])

    if has_nn:
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        _plot_3d(ax2, r_nn, "NN адаптивная V*", colors["nn"])

    fig.suptitle(f"3D траектории — {curve_label}", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, "compare_3d.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  3D траектории: {p}")

    # -----------------------------------------------------------------------
    # 4. Ошибка s_arc - V*t (синхронизация)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_base, r_base.errors[:, 0], color=colors["base"],
            linewidth=1.8, label=_label("base"))
    if has_nn:
        ax.plot(t_nn, r_nn.errors[:, 0], color=colors["nn"],
                linewidth=1.8, linestyle="--", label=_label("nn"))
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("t, с"); ax.set_ylabel("s_arc - V*t, м")
    ax.set_title(f"Ошибка синхронизации — {curve_label}")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(out_dir, "compare_sync_error.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Синхронизация: {p}")


def print_comparison_table(
    r_base: SimResult,
    r_nn: SimResult | None,
    Vstar_base: float,
) -> None:
    """Таблица итоговых метрик сравнения."""
    def _metrics(r: SimResult, name: str) -> dict:
        e1 = r.errors[:, 1]
        e2 = r.errors[:, 2]
        v  = r.velocity
        return {
            "name":        name,
            "e2_final":    float(e2[-1]),
            "e2_max":      float(np.max(np.abs(e2))),
            "e2_rms":      float(np.sqrt(np.mean(e2**2))),
            "e1_rms":      float(np.sqrt(np.mean(e1**2))),
            "v_mean":      float(np.mean(v)),
            "v_final":     float(v[-1]),
        }

    rows = [_metrics(r_base, f"Константная V*={Vstar_base}")]
    if r_nn is not None:
        rows.append(_metrics(r_nn, "NN адаптивная V*"))

    col_w = 28
    print(f"\n{'='*70}")
    print(f"  СРАВНЕНИЕ СИМУЛЯЦИЙ")
    print(f"{'='*70}")
    header = f"  {'Метрика':<22}" + "".join(f"  {r['name'][:col_w]:<{col_w}}" for r in rows)
    print(header)
    print(f"  {'-'*66}")

    metrics_info = [
        ("e2_final",  "e2 финальное, м"),
        ("e2_max",    "e2 максимальное, м"),
        ("e2_rms",    "e2 RMS, м"),
        ("e1_rms",    "e1 RMS, м"),
        ("v_mean",    "||v|| среднее, м/с"),
        ("v_final",   "||v|| финальное, м/с"),
    ]
    for key, label in metrics_info:
        line = f"  {label:<22}"
        for r in rows:
            line += f"  {r[key]:>{col_w}.4f}"
        print(line)

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Симуляция дрона: константная V* vs NN-оптимизатор",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="auto",
        help="Путь к .pt модели, 'auto' (найти в ml/data/) или 'none'"
    )
    parser.add_argument(
        "--curve", default="spiral",
        choices=["spiral", "line", "circle"],
        help="Кривая для симуляции"
    )
    parser.add_argument(
        "--Vstar", type=float, default=1.0,
        help="Базовая скорость V* для сравнения (и fallback для NN)"
    )
    parser.add_argument(
        "--no-nn", action="store_true",
        help="Запустить только baseline без NN"
    )
    parser.add_argument(
        "--compare", action="store_true", default=True,
        help="Сравнительный режим: запустить оба варианта (по умолчанию)"
    )
    parser.add_argument(
        "--out", default=_DEFAULT_OUT,
        help="Директория для графиков"
    )
    parser.add_argument(
        "--max-speed", type=float, default=3.0,
        help="max_speed дрона"
    )
    parser.add_argument(
        "--min-speed", type=float, default=0.3,
        help="min_speed дрона"
    )
    args = parser.parse_args()

    ensure_out(args.out)
    drone = QuadModel(max_speed=args.max_speed, min_speed=args.min_speed)
    curve, cfg_kw, curve_label = make_scenario_curve(args.curve)

    # --- Определить, запускать ли NN ---
    run_nn = not args.no_nn
    model_path = None
    speed_fn = None

    if run_nn:
        if args.model.lower() == "none":
            run_nn = False
        else:
            model_path = resolve_model_path(args.model)
            if model_path is None:
                run_nn = False

    if run_nn:
        # Загрузка модели
        from ml.inference.predict import SpeedPredictor
        from ml.dataset.features import feature_vector

        predictor = SpeedPredictor.load(model_path, drone=drone)
        print(f"  Загружена модель: {predictor}")

        def speed_fn(state: np.ndarray, s: float) -> float:
            feat = feature_vector(state, curve, drone=drone, s=s)
            return predictor.predict(feat)

    print(f"\nКривая   : {curve_label}")
    print(f"Дрон     : V* ∈ [{drone.min_speed}, {drone.max_speed}]")
    print(f"Baseline : Vstar={args.Vstar}")
    print(f"NN       : {'включён' if run_nn else 'выключен'}\n")

    # --- Запуск baseline ---
    print("--- Baseline (константная V*) ---")
    r_base = run_simulation(curve, cfg_kw, args.Vstar, drone,
                            speed_fn=None, label="baseline")
    t_base = np.linspace(0, cfg_kw["T"], len(r_base.errors))

    # --- Запуск NN ---
    r_nn = None
    t_nn = None
    if run_nn:
        print("\n--- NN-оптимизатор ---")
        r_nn = run_simulation(curve, cfg_kw, args.Vstar, drone,
                              speed_fn=speed_fn, label="NN")
        t_nn = np.linspace(0, cfg_kw["T"], len(r_nn.errors))

    # --- Таблица метрик ---
    print_comparison_table(r_base, r_nn, args.Vstar)

    # --- Графики ---
    print("Строю графики сравнения...")
    plot_comparison(t_base, r_base, t_nn, r_nn, args.Vstar, args.out, curve_label)
    print(f"\nВсе графики сохранены в: {args.out}")


if __name__ == "__main__":
    main()
