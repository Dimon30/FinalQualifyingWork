"""
Тестовый запуск дрона вдоль произвольной кривой.

Кривая: эллиптическая спираль  p(s) = [4·cos(s), 2·sin(s), 0.5·s]
Параметры: нормализованная модель (mass=1, J=1), V*=1.0, T=40с, dt=0.002

По умолчанию — базовая симуляция с константной V*.
При указании --model — загружается нейросетевой оптимизатор SpeedPredictor,
который адаптивно выбирает V* на каждом шаге симуляции.

Запуск (из корня проекта):
    python code/scenarios/run_test_drone.py
    python code/scenarios/run_test_drone.py --model auto
    python code/scenarios/run_test_drone.py --model default
    python code/scenarios/run_test_drone.py --model code/ml/data/saved_models/speed_model.pt
    python code/scenarios/run_test_drone.py --out code/out_images/my_run
"""
from __future__ import annotations

import argparse
import os
import sys

# Добавить code/ в sys.path для импорта drone_sim и ml.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")

from drone_sim import make_curve, SimConfig, QuadModel, simulate_path_following
from drone_sim.visualization.plotting import ensure_out, display_path

# ---------------------------------------------------------------------------
# Пути по умолчанию
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_DEFAULT_OUT = "code/out_images/test_drone"
_DEFAULT_MODEL_PATH = "code/ml/data/saved_models/speed_model.pt"

# ---------------------------------------------------------------------------
# Кривая: эллиптическая спираль
# Примечание: ||t||² ∈ [4.25, 16.25], поэтому gamma_nearest = 5
# (условие: gamma < 2 / (||t||²_max * dt) = 2 / (16.25 * 0.002) ≈ 61.5)
# ---------------------------------------------------------------------------
def _make_curve():
    return make_curve(lambda s: np.array([4.0 * np.cos(s), 2.0 * np.sin(s), 0.5 * s]))


def _make_x0() -> np.ndarray:
    x0 = np.zeros(16)
    x0[0:3] = np.array([4.0, 0.0, 0.0])   # Начальное положение на кривой (s=0).
    return x0


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _resolve_model(arg: str) -> str | None:
    """Разрешить аргумент --model в путь к файлу модели.

    Допустимые значения:
        'none'    — не использовать NN
        'default' — стандартный путь проекта (code/ml/data/saved_models/speed_model.pt)
        'auto'    — найти последний .pt в code/ml/data/
        иное      — прямой путь к файлу
    """
    if arg.lower() == "none":
        return None

    if arg.lower() in ("default", "auto"):
        if arg.lower() == "default":
            path = _DEFAULT_MODEL_PATH
            if os.path.isfile(path):
                return path
            print(f"  [ПРЕДУПРЕЖДЕНИЕ] Модель не найдена: {path}")
            return None
        # auto: найти самый свежий .pt в code/ml/data/
        search_dir = os.path.join(_HERE, "..", "ml", "data")
        candidates = []
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.endswith(".pt"):
                    p = os.path.join(root, f)
                    candidates.append((os.path.getmtime(p), p))
        if not candidates:
            print("  [ПРЕДУПРЕЖДЕНИЕ] Файлы .pt не найдены в code/ml/data/. Запуск без NN.")
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
    """Загрузить SpeedPredictor и вернуть speed_fn для SimConfig.

    Возвращает (speed_fn, drone) или (None, QuadModel()) при ошибке.
    """
    try:
        from ml.inference.predict import SpeedPredictor
        from ml.dataset.features import feature_vector
    except ImportError as e:
        print(f"  [ПРЕДУПРЕЖДЕНИЕ] Не удалось импортировать ML-модуль: {e}")
        print("  Установите зависимости: pip install torch")
        return None, QuadModel()

    predictor = SpeedPredictor.load(model_path)
    drone = predictor.drone

    curve_ref = _make_curve()   # Кривая нужна внутри замыкания.

    def speed_fn(state: np.ndarray, s: float) -> float:
        feat = feature_vector(state, curve_ref, drone=drone, s=s)
        return predictor.predict(feat)

    return speed_fn, drone


# ---------------------------------------------------------------------------
# Основная функция запуска
# ---------------------------------------------------------------------------

def run(
    out_dir: str = _DEFAULT_OUT,
    Vstar: float = 1.0,
    T: float = 40.0,
    model_path: str | None = None,
) -> None:
    """Запустить симуляцию дрона вдоль эллиптической спирали.

    Параметры:
        out_dir    — директория для сохранения графиков
        Vstar      — базовая параметрическая скорость (используется если нет NN)
        T          — время симуляции, сек
        model_path — путь к .pt файлу SpeedPredictor; None → без NN
    """
    ensure_out(out_dir)

    curve = _make_curve()
    x0 = _make_x0()

    # Определить speed_fn и drone.
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Тестовый запуск дрона вдоль эллиптической спирали",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="none",
        metavar="PATH|auto|default|none",
        help=(
            "Нейросетевой оптимизатор V*: путь к .pt файлу, "
            "'auto' (последний в code/ml/data/), "
            "'default' (стандартный путь проекта), "
            "'none' (не использовать NN)"
        ),
    )
    parser.add_argument(
        "--out", default=_DEFAULT_OUT,
        help="Директория для сохранения графиков (относительно корня проекта)",
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
