"""BenchmarkRunner — запуск всех моделей на фиксированном наборе кривых.

Использование::

    from ml.evaluation.test_suite import get_test_suite
    from ml.evaluation.benchmark import BenchmarkRunner

    runner = BenchmarkRunner(
        model_paths={
            "mlp": "code_app/ml/data/saved_models/speed_model.pt",
            "sac": "code_app/ml/data/saved_models/sac_model.pt",
            "td3": "code_app/ml/data/saved_models/td3_model.pt",
            "ppo": "code_app/ml/data/saved_models/ppo_model.pt",
        }
    )
    results = runner.run(get_test_suite())
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from drone_sim import SimConfig, simulate_path_following
from drone_sim.models.quad_model import QuadModel
from ml.evaluation.test_suite import TestScenario
from ml.models.registry import SpeedPredictorAny
from ml.dataset.features import feature_vector


# ---------------------------------------------------------------------------
# Результат одного прогона (модель × кривая)
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Метрики одного прогона симуляции.

    Атрибуты:
        model_name    — 'baseline', 'mlp', 'sac', 'td3', 'ppo'
        scenario_name — имя сценария (например 'spiral_r3')
        e1_rms        — RMS тангенциальной ошибки [м]
        e2_rms        — RMS поперечной ошибки [м]
        e2_max        — максимальная |e2| [м]
        v_mean        — средняя скорость [м/с]
        v_final       — финальная скорость [м/с]
        speedup       — v_mean / baseline_v_mean (заполняется после baseline)
        converged     — True если e2_rms < CONVERGE_THRESH
        t             — массив времени [n]
        errors        — [n×4]: [s_arc-s_ref, e1, e2, delta_phi]
        velocity      — [n]: ||v||
    """
    model_name: str
    scenario_name: str
    e1_rms: float
    e2_rms: float
    e2_max: float
    v_mean: float
    v_final: float
    speedup: float
    converged: bool
    t: np.ndarray
    errors: np.ndarray
    velocity: np.ndarray


CONVERGE_THRESH = 0.1   # м — порог сходимости e2_rms


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Запускает все модели на всех тестовых сценариях и собирает метрики.

    Параметры:
        model_paths — словарь {кодовое_имя → путь_к_.pt}.
                      Если модель не найдена/не загружается — пропускается.
        Vstar_base  — базовая V* для режима «baseline» (константная скорость).
        verbose     — печатать прогресс.
    """

    def __init__(
        self,
        model_paths: dict[str, str],
        Vstar_base: float = 1.0,
        verbose: bool = True,
    ) -> None:
        self._paths = model_paths
        self._Vstar_base = float(Vstar_base)
        self._verbose = verbose

        # Загружаем предикторы (пропускаем битые/отсутствующие)
        self._predictors: dict[str, SpeedPredictorAny] = {}
        for name, path in model_paths.items():
            try:
                pred = SpeedPredictorAny.load(path)
                self._predictors[name] = pred
                if verbose:
                    print(f"  [OK] {name:6s}  {path}")
            except Exception as exc:
                warnings.warn(
                    f"Модель '{name}' не загружена ({path}): {exc}",
                    UserWarning,
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Публичный метод
    # ------------------------------------------------------------------

    def run(
        self,
        suite: list[TestScenario],
    ) -> list[ModelResult]:
        """Запустить бенчмарк по всем сценариям.

        Для каждого сценария:
          1. Прогон baseline (константная V*)
          2. Прогон каждой загруженной модели (адаптивная V*)

        Возвращает список ModelResult (один объект на прогон).
        """
        all_results: list[ModelResult] = []

        for sc in suite:
            if self._verbose:
                print(f"\n{'='*60}")
                print(f"Сценарий: {sc.name}  ({sc.label})")
                print(f"{'='*60}")

            # ---- Baseline ------------------------------------------------
            r_base = self._run_one(sc, model_name="baseline", speed_fn=None)
            all_results.append(r_base)

            # ---- Модели --------------------------------------------------
            for name, pred in self._predictors.items():
                speed_fn = self._make_speed_fn(pred, sc)
                r = self._run_one(sc, model_name=name, speed_fn=speed_fn)
                # speedup относительно baseline
                r.speedup = r.v_mean / r_base.v_mean if r_base.v_mean > 1e-6 else 1.0
                all_results.append(r)

        return all_results

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _run_one(
        self,
        sc: TestScenario,
        model_name: str,
        speed_fn,
    ) -> ModelResult:
        """Запустить одну симуляцию и вернуть ModelResult."""
        drone = QuadModel()

        cfg_kw = dict(sc.cfg_kw)
        cfg = SimConfig(
            Vstar=self._Vstar_base,
            quad_model=drone,
            speed_fn=speed_fn,
            warmup_time=sc.warmup_time if speed_fn is not None else 0.0,
            vstar_max_rate=sc.vstar_rate,
            x0=sc.x0.copy(),
            **cfg_kw,
        )

        if self._verbose:
            tag = f"[{model_name}]"
            print(f"  {tag:<12}", end="", flush=True)

        result = simulate_path_following(sc.curve, cfg)

        e1 = result.errors[:, 1]
        e2 = result.errors[:, 2]
        v  = result.velocity
        T  = cfg.T

        e1_rms = float(np.sqrt(np.mean(e1 ** 2)))
        e2_rms = float(np.sqrt(np.mean(e2 ** 2)))
        e2_max = float(np.max(np.abs(e2)))
        v_mean = float(np.mean(v))
        v_fin  = float(v[-1])

        t_arr  = np.linspace(0.0, T, len(result.errors))

        if self._verbose:
            print(
                f"e1_rms={e1_rms:.4f}  e2_rms={e2_rms:.4f}  "
                f"e2_max={e2_max:.4f}  v_mean={v_mean:.3f} м/с"
            )

        return ModelResult(
            model_name=model_name,
            scenario_name=sc.name,
            e1_rms=e1_rms,
            e2_rms=e2_rms,
            e2_max=e2_max,
            v_mean=v_mean,
            v_final=v_fin,
            speedup=1.0,       # будет перезаписан для NN-моделей
            converged=e2_rms < CONVERGE_THRESH,
            t=t_arr,
            errors=result.errors,
            velocity=v,
        )

    def _make_speed_fn(self, pred: SpeedPredictorAny, sc: TestScenario):
        """Замыкание: speed_fn(state, s) → V* из дрона [min_speed, max_speed]."""
        drone = pred.drone
        curve = sc.curve

        def speed_fn(state: np.ndarray, s: float) -> float:
            feat = feature_vector(state, curve, drone=drone, s=s)
            return pred.predict(feat)

        return speed_fn
