"""
Сравнение: базовая симуляция (константная V*) vs нейросетевой оптимизатор SpeedPredictor.

Запускает оба варианта подряд на выбранной кривой и строит сравнительные графики.
Для запуска одной симуляции с NN используйте run_test_drone.py.

Запуск (из корня проекта):
    python code/scenarios/run_nn_speed.py
    python code/scenarios/run_nn_speed.py --curve spiral
    python code/scenarios/run_nn_speed.py --curve line --Vstar 1.0
    python code/scenarios/run_nn_speed.py --model code/ml/data/saved_models/speed_model.pt
    python code/scenarios/run_nn_speed.py --out code/out_images/compare_spiral
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_sim import make_curve, SimConfig, simulate_path_following
from drone_sim.geometry.curves import spiral_curve, line_xyz_curve, nearest_point_line, CurveGeom
from drone_sim.models.quad_model import QuadModel
from drone_sim.simulation.path_sim import SimResult
from drone_sim.visualization.plotting import ensure_out, display_path

# ---------------------------------------------------------------------------
# Пути по умолчанию
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_DEFAULT_MODEL = os.path.join(_HERE, "..", "ml", "data", "saved_models", "speed_model.pt")
_DEFAULT_OUT = "code/out_images/nn_speed"


# ---------------------------------------------------------------------------
# Конфигурация кривых
# ---------------------------------------------------------------------------

def _make_scenario(name: str) -> tuple[CurveGeom, dict, str]:
    """Вернуть (curve, cfg_kwargs, label) для заданного имени кривой."""
    if name == "spiral":
        curve = spiral_curve(r=3.0)
        x0 = np.zeros(16); x0[0:3] = [2.9, 0.0, 0.0]
        cfg_kw = dict(T=40.0, dt=0.002, kappa=200.0,
                      gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0, x0=x0)
        label = "Спираль r=3"
    elif name == "line":
        curve = line_xyz_curve()
        x0 = np.zeros(16); x0[0:3] = [1.0, 1.0, 0.0]
        cfg_kw = dict(T=30.0, dt=0.005, kappa=100.0,
                      gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0,
                      nearest_fn=nearest_point_line, x0=x0)
        label = "Прямая x=s,y=s,z=s"
    elif name == "circle":
        curve = make_curve(lambda s: np.array([3.0 * np.cos(s / 3.0),
                                                3.0 * np.sin(s / 3.0), 0.0]))
        x0 = np.zeros(16); x0[0:3] = [2.9, 0.0, 0.0]
        cfg_kw = dict(T=40.0, dt=0.002, kappa=200.0,
                      gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0, x0=x0)
        label = "Окружность r=3"
    else:
        raise ValueError(f"Неизвестная кривая: {name!r}. Допустимо: spiral, line, circle")
    return curve, cfg_kw, label


# ---------------------------------------------------------------------------
# Загрузка NN модели
# ---------------------------------------------------------------------------

def _load_speed_fn(model_path: str, curve: CurveGeom):
    """Загрузить SpeedPredictor и вернуть (speed_fn, drone)."""
    from ml.inference.predict import SpeedPredictor
    from ml.dataset.features import feature_vector

    predictor = SpeedPredictor.load(model_path)
    drone = predictor.drone

    def speed_fn(state: np.ndarray, s: float) -> float:
        feat = feature_vector(state, curve, drone=drone, s=s)
        return predictor.predict(feat)

    return speed_fn, drone, predictor


# ---------------------------------------------------------------------------
# Запуск симуляции
# ---------------------------------------------------------------------------

def _run(curve: CurveGeom, cfg_kw: dict, Vstar: float,
         drone: QuadModel, speed_fn=None, label: str = "") -> SimResult:
    cfg = SimConfig(Vstar=Vstar, quad_model=drone, speed_fn=speed_fn, **cfg_kw)
    print(f"  [{label}]  T={cfg.T}с  dt={cfg.dt}  kappa={cfg.kappa}", end="", flush=True)
    result = simulate_path_following(curve, cfg)
    print(f"  → e2_final={result.errors[-1, 2]:+.4f}  vel={result.velocity[-1]:.3f} м/с")
    return result


# ---------------------------------------------------------------------------
# Сравнительные графики
# ---------------------------------------------------------------------------

def _plot_comparison(
    t_base: np.ndarray, r_base: SimResult,
    t_nn: np.ndarray, r_nn: SimResult,
    Vstar_base: float, out_dir: str, label: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    C = {"base": (0.0078, 0.447, 0.741), "nn": (0.85, 0.33, 0.10), "ref": (0.5, 0.5, 0.5)}

    # Ошибки e1, e2.
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for i, (ax, col, ylabel, title) in enumerate(zip(
        axes,
        [1, 2],
        ["e1, м", "e2, м"],
        [f"Тангенциальная ошибка e1 — {label}", "Поперечная ошибка e2"],
    )):
        ax.plot(t_base, r_base.errors[:, col], color=C["base"], linewidth=1.8,
                label="Константная V*")
        ax.plot(t_nn, r_nn.errors[:, col], color=C["nn"], linewidth=1.8,
                linestyle="--", label="NN V*")
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    axes[1].set_xlabel("t, с")
    fig.tight_layout()
    p = os.path.join(out_dir, "compare_errors.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Ошибки       : {display_path(p)}")

    # Скорости.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_base, r_base.velocity, color=C["base"], linewidth=1.8,
            label=f"Константная V*={Vstar_base}")
    ax.plot(t_nn, r_nn.velocity, color=C["nn"], linewidth=1.8,
            linestyle="--", label="NN (адаптивная V*)")
    ax.axhline(Vstar_base, color=C["ref"], linestyle=":", linewidth=1.5,
               label=f"Базовая V*={Vstar_base}")
    ax.set_xlabel("t, с"); ax.set_ylabel("||v||, м/с")
    ax.set_title(f"Линейная скорость — {label}")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(out_dir, "compare_velocity.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Скорости     : {display_path(p)}")

    # 3D траектории.
    fig = plt.figure(figsize=(14, 6))
    for i, (result, title, color) in enumerate([
        (r_base, f"Константная V*={Vstar_base}", C["base"]),
        (r_nn, "NN адаптивная V*", C["nn"]),
    ]):
        ax3 = fig.add_subplot(1, 2, i + 1, projection="3d")
        p_ref = np.stack([result.curve.p(z) for z in result.zeta], axis=0)
        ax3.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2],
                 "--", color=C["ref"], linewidth=1.2, label="Заданная")
        ax3.plot(result.x[:, 0], result.x[:, 1], result.x[:, 2],
                 color=color, linewidth=1.8, label="Дрон")
        ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
        ax3.set_title(title); ax3.legend(fontsize=8)
    fig.suptitle(f"3D траектории — {label}", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, "compare_3d.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  3D траектории: {display_path(p)}")

    # Ошибка синхронизации s_arc - V*t.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_base, r_base.errors[:, 0], color=C["base"], linewidth=1.8,
            label="Константная V*")
    ax.plot(t_nn, r_nn.errors[:, 0], color=C["nn"], linewidth=1.8,
            linestyle="--", label="NN V*")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("t, с"); ax.set_ylabel("s_arc - V*t, м")
    ax.set_title(f"Ошибка синхронизации — {label}")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(out_dir, "compare_sync_error.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Синхронизация: {display_path(p)}")


def _print_table(r_base: SimResult, r_nn: SimResult, Vstar: float) -> None:
    def _m(r: SimResult, name: str) -> dict:
        e1, e2, v = r.errors[:, 1], r.errors[:, 2], r.velocity
        return dict(name=name,
                    e2_final=float(e2[-1]), e2_max=float(np.max(np.abs(e2))),
                    e2_rms=float(np.sqrt(np.mean(e2**2))),
                    e1_rms=float(np.sqrt(np.mean(e1**2))),
                    v_mean=float(np.mean(v)), v_final=float(v[-1]))

    rows = [_m(r_base, f"Константная V*={Vstar}"), _m(r_nn, "NN адаптивная V*")]
    w = 28
    sep = "=" * 70
    print(f"\n{sep}")
    print("  СРАВНЕНИЕ СИМУЛЯЦИЙ")
    print(sep)
    print(f"  {'Метрика':<22}" + "".join(f"  {r['name'][:w]:<{w}}" for r in rows))
    print(f"  {'-'*66}")
    for key, lbl in [
        ("e2_final", "e2 финальное, м"), ("e2_max", "e2 максимальное, м"),
        ("e2_rms", "e2 RMS, м"), ("e1_rms", "e1 RMS, м"),
        ("v_mean", "||v|| среднее, м/с"), ("v_final", "||v|| финальное, м/с"),
    ]:
        print(f"  {lbl:<22}" + "".join(f"  {r[key]:{w}.4f}" for r in rows))
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнение: константная V* vs нейросетевой оптимизатор SpeedPredictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="auto",
                        help="Путь к .pt модели или 'auto' (найти в code/ml/data/)")
    parser.add_argument("--curve", default="spiral",
                        choices=["spiral", "line", "circle"],
                        help="Кривая для сравнения")
    parser.add_argument("--Vstar", type=float, default=1.0,
                        help="Базовая V* для константного режима")
    parser.add_argument("--out", default=_DEFAULT_OUT,
                        help="Директория для сравнительных графиков")
    args = parser.parse_args()

    # Разрешить путь к модели.
    model_path = args.model
    if model_path == "auto":
        search = os.path.join(_HERE, "..", "ml", "data")
        candidates = []
        for root, _, files in os.walk(search):
            for f in files:
                if f.endswith(".pt"):
                    p = os.path.join(root, f)
                    candidates.append((os.path.getmtime(p), p))
        if not candidates:
            print("  [ОШИБКА] Файлы .pt не найдены в code/ml/data/.")
            print("  Сначала запустите обучение: python code/scenarios/train_speed_model.py")
            sys.exit(1)
        candidates.sort(reverse=True)
        model_path = candidates[0][1]
        print(f"  Авто-выбор модели: {model_path}")
    elif not os.path.isfile(model_path):
        print(f"  [ОШИБКА] Файл модели не найден: {model_path}")
        sys.exit(1)

    ensure_out(args.out)
    curve, cfg_kw, curve_label = _make_scenario(args.curve)

    # Загрузить NN.
    speed_fn, drone, predictor = _load_speed_fn(model_path, curve)
    print(f"\nМодель  : {predictor}")
    print(f"Кривая  : {curve_label}")
    print(f"Дрон    : V* ∈ [{drone.min_speed}, {drone.max_speed}]  "
          f"lateral_e_lim={drone.lateral_error_limit}")
    print(f"Baseline: Vstar={args.Vstar}\n")

    # Baseline.
    print("--- Baseline (константная V*) ---")
    r_base = _run(curve, cfg_kw, args.Vstar, drone, speed_fn=None, label="baseline")
    t_base = np.linspace(0, cfg_kw["T"], len(r_base.errors))

    # NN.
    print("\n--- NN-оптимизатор ---")
    r_nn = _run(curve, cfg_kw, args.Vstar, drone, speed_fn=speed_fn, label="NN")
    t_nn = np.linspace(0, cfg_kw["T"], len(r_nn.errors))

    # Таблица метрик.
    _print_table(r_base, r_nn, args.Vstar)

    # Сравнительные графики.
    print("Строю сравнительные графики...")
    _plot_comparison(t_base, r_base, t_nn, r_nn, args.Vstar, args.out, curve_label)
    print(f"\nВсе графики сохранены в: {display_path(args.out)}")


if __name__ == "__main__":
    main()
