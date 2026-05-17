"""Тестовый запуск дрона вдоль эллиптической спирали p(s) = [4·cos(s), 2·sin(s), 0.5·s].

По умолчанию — константная V*; с --model подключается NN-оптимизатор V*.

Запуск из корня проекта:
    python code_app/scenarios/run_test_drone.py
    python code_app/scenarios/run_test_drone.py --model auto
    python code_app/scenarios/run_test_drone.py --model default
    python code_app/scenarios/run_test_drone.py --out code_app/out_images/my_run
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

from drone_sim import make_curve, SimConfig, QuadModel, simulate_path_following
from drone_sim.visualization.plotting import ensure_out, display_path

_HERE = os.path.dirname(__file__)
_DEFAULT_OUT = "code_app/out_images/test_drone"
_DEFAULT_MODEL_PATH = "code_app/ml/data/saved_models/speed_model.pt"


def _make_curve():
    """Эллиптическая спираль: ||t||² ∈ [4.25, 16.25], gamma_nearest=5 безопасен при dt=0.002."""
    return make_curve(lambda s: np.array([3.0 * np.cos(s), 3.0 * np.sin(s), 0.5 * s]))


def _make_x0() -> np.ndarray:
    x0 = np.zeros(16)
    x0[0:3] = np.array([4.0, 0.0, 0.0])
    return x0


def _resolve_model(arg: str) -> str | None:
    """Разрешить --model в путь к .pt. Допустимо: none | default | auto | <path>."""
    if arg.lower() == "none":
        return None

    if arg.lower() == "default":
        path = _DEFAULT_MODEL_PATH
        if os.path.isfile(path):
            return path
        print(f"  [ПРЕДУПРЕЖДЕНИЕ] Модель не найдена: {path}")
        return None

    if arg.lower() == "auto":
        search_dir = os.path.join(_HERE, "..", "ml", "data")
        candidates = []
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.endswith(".pt"):
                    p = os.path.join(root, f)
                    candidates.append((os.path.getmtime(p), p))
        if not candidates:
            print("  [ПРЕДУПРЕЖДЕНИЕ] Файлы .pt не найдены в code_app/ml/data/. Запуск без NN.")
            return None
        candidates.sort(reverse=True)
        found = candidates[0][1]
        print(f"  Авто-выбор модели: {found}")
        return found

    if os.path.isfile(arg):
        return arg

    print(f"  [ПРЕДУПРЕЖДЕНИЕ] Файл модели не найден: {arg}. Запуск без NN.")
    return None


def _load_speed_fn(model_path: str):
    """Загрузить SpeedPredictorAny; вернуть (speed_fn, drone). При ошибке — (None, QuadModel())."""
    try:
        from ml.models.registry import SpeedPredictorAny
        from ml.dataset.features import feature_vector
    except ImportError as e:
        print(f"  [ПРЕДУПРЕЖДЕНИЕ] Не удалось импортировать ML-модуль: {e}")
        print("  Установите зависимости: pip install torch")
        return None, QuadModel()

    predictor = SpeedPredictorAny.load(model_path)
    drone = predictor.drone

    curve_ref = _make_curve()

    def speed_fn(state: np.ndarray, s: float) -> float:
        feat = feature_vector(state, curve_ref, drone=drone, s=s)
        return predictor.predict(feat)

    return speed_fn, drone


def run(
    out_dir: str = _DEFAULT_OUT,
    Vstar: float = 1.0,
    T: float = 40.0,
    model_path: str | None = None,
) -> None:
    """Симуляция эллиптической спирали; если model_path задан — NN-оптимизатор V*."""
    ensure_out(out_dir)

    curve = _make_curve()
    x0 = _make_x0()

    speed_fn = None
    drone = QuadModel()

    if model_path is not None:
        speed_fn, drone = _load_speed_fn(model_path)
        if speed_fn is None:
            print("  Продолжаю без NN.")

    cfg = SimConfig(
        quad_model=drone,
        Vstar=Vstar,
        T=T,
        dt=0.002,
        x0=x0,
        kappa=200.0,
        gamma=(1., 3., 5., 3., 1.),
        gamma_nearest=5.0,
        zeta0=0.0,
        speed_fn=speed_fn,
    )

    mode = "NN-оптимизатор" if speed_fn is not None else "константная V*"
    print(f"\nСимуляция: эллиптическая спираль  [{mode}]")
    print(f"  V* = {Vstar}  T = {T} с  kappa = {cfg.kappa}  dt = {cfg.dt}")
    if speed_fn is not None:
        print(f"  Модель : {model_path}")
        print(f"  Дрон   : V* ∈ [{drone.min_speed}, {drone.max_speed}]  "
              f"lateral_e_lim = {drone.lateral_error_limit}")

    result = simulate_path_following(curve, cfg)
    result.print_summary()
    result.plot(out_dir=out_dir, prefix="elliptic")
    print(f"\nГрафики сохранены в: {display_path(out_dir)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Тестовый запуск дрона вдоль эллиптической спирали",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="none",
        metavar="PATH|auto|default|none",
        help="NN-оптимизатор V*: путь к .pt | 'auto' | 'default' | 'none'",
    )
    parser.add_argument(
        "--out", default=_DEFAULT_OUT,
        help="Директория для графиков",
    )
    parser.add_argument(
        "--vstar", type=float, default=1.0,
        help="Базовая параметрическая скорость V*",
    )
    parser.add_argument(
        "--T", type=float, default=40.0,
        help="Время симуляции, сек",
    )
    args = parser.parse_args()

    model_path = _resolve_model(args.model)

    run(
        out_dir=args.out,
        Vstar=args.vstar,
        T=args.T,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()
