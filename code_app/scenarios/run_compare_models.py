"""Сравнение любой обученной модели (mlp | sac | td3 | ppo) с константной V*.

Использование (из корня проекта)::

    # Автопоиск последней модели по кодовому имени:
    python code/scenarios/run_compare_models.py --model sac --curve spiral
    python code/scenarios/run_compare_models.py --model td3 --curve line
    python code/scenarios/run_compare_models.py --model ppo --curve circle

    # Явный путь к .pt файлу:
    python code/scenarios/run_compare_models.py \\
        --model code/ml/data/saved_models/sac_model.pt --curve spiral

    # Дополнительные параметры:
    python code/scenarios/run_compare_models.py --model sac --curve spiral \\
        --Vstar 1.0 --vstar-cap 3.5 --vstar-rate 0.3 --warmup 5.0

Кодовые имена моделей:
    mlp  — SpeedMLP        (supervised MSE)
    sac  — SpeedSAC        (Gaussian NLL + entropy + twin critics)
    td3  — SpeedTD3        (BC + Q-guided actor + Polyak targets)
    ppo  — SpeedPPO        (clipped surrogate + value + entropy)
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
from ml.models.registry import SpeedPredictorAny, MODEL_NAMES

# ---------------------------------------------------------------------------
# Пути по умолчанию
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_MODELS_DIR = os.path.join(_HERE, "..", "ml", "data", "saved_models")
_DEFAULT_OUT = "code/out_images/compare_rl"

_CODE_NAMES = MODEL_NAMES  # ['mlp', 'sac', 'td3', 'ppo']


def _resolve_model_path(model_arg: str) -> str:
    """Разрешить путь к .pt файлу.

    Если ``model_arg`` — кодовое имя ('sac', 'td3', 'ppo', 'mlp'),
    ищем ``<кодовое_имя>_model.pt`` в директории сохранённых моделей.
    Если файл не найден — поднимаем понятную ошибку с подсказкой.
    """
    if model_arg.endswith(".pt") or os.sep in model_arg or "/" in model_arg:
        if not os.path.isfile(model_arg):
            print(f"  [ОШИБКА] Файл модели не найден: {model_arg}")
            sys.exit(1)
        return model_arg

    name = model_arg.lower().strip()
    if name not in _CODE_NAMES:
        print(f"  [ОШИБКА] Неизвестное кодовое имя: {name!r}.")
        print(f"  Доступные: {_CODE_NAMES}")
        sys.exit(1)

    # Для MLP ищем оба варианта имени.
    candidates = [
        os.path.join(_MODELS_DIR, f"{name}_model.pt"),
        os.path.join(_MODELS_DIR, "speed_model.pt"),  # старый MLP
    ] if name == "mlp" else [
        os.path.join(_MODELS_DIR, f"{name}_model.pt"),
    ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    print(f"  [ОШИБКА] Модель '{name}' не найдена.")
    print(f"  Ожидался файл: {os.path.join(_MODELS_DIR, name + '_model.pt')}")
    print(f"  Сначала обучите: python code/scenarios/train_rl_model.py --model {name}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Конфигурация кривых
# ---------------------------------------------------------------------------

def _make_scenario(name: str) -> tuple[CurveGeom, dict, str]:
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
# Загрузка NN-предиктора
# ---------------------------------------------------------------------------

def _load_speed_fn(
    model_path: str,
    curve: CurveGeom,
    vstar_cap: float | None = None,
):
    """Загрузить SpeedPredictorAny и вернуть (speed_fn, drone, predictor)."""
    from ml.dataset.features import feature_vector

    predictor = SpeedPredictorAny.load(model_path)
    drone = predictor.drone

    def speed_fn(state: np.ndarray, s: float) -> float:
        feat = feature_vector(state, curve, drone=drone, s=s)
        v = predictor.predict(feat)
        if vstar_cap is not None:
            v = min(v, vstar_cap)
        return v

    return speed_fn, drone, predictor


# ---------------------------------------------------------------------------
# Запуск симуляции
# ---------------------------------------------------------------------------

def _run(
    curve: CurveGeom,
    cfg_kw: dict,
    Vstar: float,
    drone: QuadModel,
    speed_fn=None,
    label: str = "",
    warmup_time: float = 5.0,
    vstar_max_rate: float = 0.5,
) -> SimResult:
    cfg = SimConfig(
        Vstar=Vstar, quad_model=drone, speed_fn=speed_fn,
        warmup_time=warmup_time, vstar_max_rate=vstar_max_rate,
        **cfg_kw,
    )
    print(f"  [{label}]  T={cfg.T}с  dt={cfg.dt}  kappa={cfg.kappa}", end="", flush=True)
    result = simulate_path_following(curve, cfg)
    print(f"  → e2_final={result.errors[-1, 2]:+.4f}  vel={result.velocity[-1]:.3f} м/с")
    return result


# ---------------------------------------------------------------------------
# Таблица метрик
# ---------------------------------------------------------------------------

def _print_table(r_base: SimResult, r_nn: SimResult, Vstar: float, model_type: str) -> None:
    def _metrics(r: SimResult, name: str) -> dict:
        e1 = r.errors[:, 1]
        e2 = r.errors[:, 2]
        v  = r.velocity
        return dict(
            name=name,
            e2_final=float(e2[-1]),
            e2_max=float(np.max(np.abs(e2))),
            e2_rms=float(np.sqrt(np.mean(e2 ** 2))),
            e1_rms=float(np.sqrt(np.mean(e1 ** 2))),
            v_mean=float(np.mean(v)),
            v_final=float(v[-1]),
        )

    rows = [
        _metrics(r_base, f"Константная V*={Vstar}"),
        _metrics(r_nn,   f"{model_type.upper()} адаптивная"),
    ]
    w = 26
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  СРАВНЕНИЕ  —  {model_type.upper()} vs константная V*")
    print(sep)
    print(f"  {'Метрика':<24}" + "".join(f"  {r['name'][:w]:<{w}}" for r in rows))
    print(f"  {'-'*68}")
    for key, lbl in [
        ("e2_final", "e2 финальное, м"),
        ("e2_max",   "e2 максимальное, м"),
        ("e2_rms",   "e2 RMS, м"),
        ("e1_rms",   "e1 RMS, м"),
        ("v_mean",   "||v|| среднее, м/с"),
        ("v_final",  "||v|| финальное, м/с"),
    ]:
        print(f"  {lbl:<24}" + "".join(f"  {r[key]:{w}.4f}" for r in rows))
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

def _plot_comparison(
    t_base: np.ndarray, r_base: SimResult,
    t_nn: np.ndarray, r_nn: SimResult,
    Vstar_base: float, out_dir: str, curve_label: str, model_type: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    C = {
        "base": (0.0078, 0.447, 0.741),
        "nn":   (0.85, 0.33, 0.10),
        "ref":  (0.5, 0.5, 0.5),
    }
    nn_label = f"{model_type.upper()} адаптивная"

    # Ошибки e1, e2.
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, col, ylabel, title in zip(
        axes,
        [1, 2],
        ["e1, м", "e2, м"],
        [f"Тангенциальная ошибка e1 — {curve_label}",
         f"Поперечная ошибка e2 — {curve_label}"],
    ):
        ax.plot(t_base, r_base.errors[:, col], color=C["base"], linewidth=1.8,
                label=f"Константная V*={Vstar_base}")
        ax.plot(t_nn,   r_nn.errors[:, col],   color=C["nn"],   linewidth=1.8,
                linestyle="--", label=nn_label)
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("t, с")
    fig.tight_layout()
    p = os.path.join(out_dir, f"{model_type}_errors.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Ошибки       : {display_path(p)}")

    # Скорости.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_base, r_base.velocity, color=C["base"], linewidth=1.8,
            label=f"Константная V*={Vstar_base}")
    ax.plot(t_nn,   r_nn.velocity,   color=C["nn"],   linewidth=1.8,
            linestyle="--", label=nn_label)
    ax.axhline(Vstar_base, color=C["ref"], linestyle=":", linewidth=1.5,
               label=f"Базовая V*={Vstar_base}")
    ax.set_xlabel("t, с"); ax.set_ylabel("||v||, м/с")
    ax.set_title(f"Линейная скорость — {curve_label}")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{model_type}_velocity.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Скорости     : {display_path(p)}")

    # 3D траектории.
    fig = plt.figure(figsize=(14, 6))
    for i, (result, title, color) in enumerate([
        (r_base, f"Константная V*={Vstar_base}", C["base"]),
        (r_nn,   nn_label, C["nn"]),
    ]):
        ax3 = fig.add_subplot(1, 2, i + 1, projection="3d")
        p_ref = np.stack([result.curve.p(z) for z in result.zeta], axis=0)
        ax3.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2],
                 "--", color=C["ref"], linewidth=1.2, label="Заданная")
        ax3.plot(result.x[:, 0], result.x[:, 1], result.x[:, 2],
                 color=color, linewidth=1.8, label="Дрон")
        ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
        ax3.set_title(title); ax3.legend(fontsize=8)
    fig.suptitle(f"3D траектории — {curve_label}", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{model_type}_3d.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  3D траектории: {display_path(p)}")

    # Ошибка синхронизации s_arc - V*t.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_base, r_base.errors[:, 0], color=C["base"], linewidth=1.8,
            label=f"Константная V*={Vstar_base}")
    ax.plot(t_nn,   r_nn.errors[:, 0],   color=C["nn"],   linewidth=1.8,
            linestyle="--", label=nn_label)
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("t, с"); ax.set_ylabel("s_arc - V*t, м")
    ax.set_title(f"Ошибка синхронизации — {curve_label}")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{model_type}_sync.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Синхронизация: {display_path(p)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнение обученной модели (mlp|sac|td3|ppo) с константной V*",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="sac",
        help="Кодовое имя ('mlp','sac','td3','ppo') или путь к .pt файлу",
    )
    parser.add_argument(
        "--curve", default="spiral",
        choices=["spiral", "line", "circle"],
        help="Тестовая кривая",
    )
    parser.add_argument(
        "--Vstar", type=float, default=1.0,
        help="Базовая V* для константного режима",
    )
    parser.add_argument(
        "--vstar-cap", type=float, default=None, metavar="CAP",
        help="Верхний предел NN-предсказания V* (None = без ограничения). "
             "Для spiral рекомендуется 3.5.",
    )
    parser.add_argument(
        "--warmup", type=float, default=5.0, metavar="SEC",
        help="Время прогрева [с]: NN неактивна, используется константная V*",
    )
    parser.add_argument(
        "--vstar-rate", type=float, default=0.5, metavar="RATE",
        help="Макс. скорость изменения V* [1/с] при NN-управлении",
    )
    parser.add_argument(
        "--out", default=_DEFAULT_OUT,
        help="Директория для сравнительных графиков",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Не строить графики",
    )
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model)
    ensure_out(args.out)
    curve, cfg_kw, curve_label = _make_scenario(args.curve)

    speed_fn, drone, predictor = _load_speed_fn(
        model_path, curve, vstar_cap=args.vstar_cap
    )
    model_type = predictor.model_type

    print(f"\nМодель    : {predictor}")
    print(f"Кривая    : {curve_label}")
    print(f"Дрон      : V* ∈ [{drone.min_speed}, {drone.max_speed}]  "
          f"lateral_e_lim={drone.lateral_error_limit}")
    print(f"Baseline  : Vstar={args.Vstar}")
    if args.vstar_cap is not None:
        print(f"NN cap    : V*_nn <= {args.vstar_cap}")
    print(f"Warmup    : {args.warmup} с   rate: {args.vstar_rate} V*/с\n")

    print("--- Baseline (константная V*) ---")
    r_base = _run(
        curve, cfg_kw, args.Vstar, drone,
        speed_fn=None, label="baseline",
        warmup_time=args.warmup, vstar_max_rate=args.vstar_rate,
    )
    t_base = np.linspace(0, cfg_kw["T"], len(r_base.errors))

    print(f"\n--- {model_type.upper()}-оптимизатор ---")
    r_nn = _run(
        curve, cfg_kw, args.Vstar, drone,
        speed_fn=speed_fn, label=model_type.upper(),
        warmup_time=args.warmup, vstar_max_rate=args.vstar_rate,
    )
    t_nn = np.linspace(0, cfg_kw["T"], len(r_nn.errors))

    _print_table(r_base, r_nn, args.Vstar, model_type)

    if not args.no_plots:
        print("Строю сравнительные графики...")
        _plot_comparison(
            t_base, r_base, t_nn, r_nn,
            args.Vstar, args.out, curve_label, model_type,
        )
        print(f"\nВсе графики сохранены в: {display_path(args.out)}")


if __name__ == "__main__":
    main()
