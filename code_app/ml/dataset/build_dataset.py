"""Dataset builder for the speed model."""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from drone_sim.models.quad_model import QuadModel
from ml.config import ORACLE_DT, ORACLE_KAPPA, OracleConfig, auto_rollout_horizon
from ml.curves.generator import generate_curve, CurveSpec
from ml.dataset.curve_generator import validate_curve
from ml.dataset.features import extract_features
from ml.dataset.simulator_wrapper import (
    find_optimal_speed,
    rollout_with_speed,
    is_stable,
)

# Logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# CSV schema.
_CSV_COLUMNS = [
    "e1", "e2", "de2_dt", "v_norm", "heading_error",
    "kappa", "kappa_max_lookahead",
    "s", "t_norm",
    "V_opt",
]

# Default parameter range.
_DEFAULT_S_START: float = 0.0
_DEFAULT_S_END: float = 15.0

# Default output path.
_DEFAULT_OUT_PATH: str = "code_app/ml/data/dataset.csv"


def _make_state_on_curve(
    spec: CurveSpec,
    s: float,
    rng: Optional[np.random.Generator] = None,
    pos_std: float = 0.05,
    vel_std: float = 0.1,
) -> np.ndarray:
    """Create a 16D state near the curve at ``s`` with optional perturbations.

    Without perturbations the drone sits exactly on the curve with zero
    velocity, which makes all dynamic features (e1, e2, v_norm, …) identically
    zero in the dataset.  Passing ``rng`` adds small Gaussian noise so that
    the dataset captures realistic non-zero deviations.

    Углы инициализируются из геометрии кривой в точке s:
        phi   = yaw_star(s)   -- рысканье вдоль касательной
        theta = beta(s)       -- тангаж вдоль касательной
    Это предотвращает большую начальную ошибку d_phi = phi - yaw_star(s),
    которая при phi=0 могла достигать O(1 рад) и делать oracle-ролауты
    нестабильными из-за transient-всплеска e2.
    """
    x0 = np.zeros(16, dtype=float)
    x0[0:3] = spec.curve.p(s)          # Position on the curve.
    x0[6] = spec.curve.yaw_star(s)     # phi aligned with tangent direction.
    x0[7] = spec.curve.beta(s)         # theta aligned with tangent pitch.
    if rng is not None:
        x0[0:3] += rng.normal(0.0, pos_std, 3)   # ±pos_std м от кривой
        x0[3:6] += rng.normal(0.0, vel_std, 3)   # ±vel_std м/с скорость
    return x0


def _build_record(
    feats: dict,
    s: float,
    t_norm: float,
    V_opt: float,
) -> dict:
    """Build one CSV record."""
    return {
        "e1": feats["e1"],
        "e2": feats["e2"],
        "de2_dt": feats["de2_dt"],
        "v_norm": feats["v_norm"],
        "heading_error": feats["heading_error"],
        "kappa": feats["kappa"],
        "kappa_max_lookahead": feats["kappa_max_lookahead"],
        "s": round(s, 6),
        "t_norm": round(t_norm, 6),
        "V_opt": round(V_opt, 6),
    }


def _save_csv(records: list[dict], out_path: str) -> None:
    """Write records to CSV."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def generate_dataset(
    num_curves: int = 20,
    out_path: str = _DEFAULT_OUT_PATH,
    seed: int = 30,
    n_samples_per_curve: int = 5,
    s_start: float = _DEFAULT_S_START,
    s_end: float = _DEFAULT_S_END,
    drone: Optional[QuadModel] = None,
    oracle_cfg: Optional[OracleConfig] = None,
    coarse_to_fine: bool = False,
    oracle_dt: float = ORACLE_DT,
    oracle_kappa: float = ORACLE_KAPPA,
) -> str:
    """Generate a dataset and write it to CSV."""
    if drone is None:
        drone = QuadModel()
    if oracle_cfg is None:
        # Вычислить горизонт ролаута автоматически из геометрии датасета:
        # дрон должен успеть пройти хотя бы одну секцию между sample-точками.
        horizon = auto_rollout_horizon(
            s_start=s_start,
            s_end=s_end,
            n_samples=n_samples_per_curve,
            min_speed=drone.min_speed,
        )
        oracle_cfg = OracleConfig(rollout_horizon=horizon)

    rng = np.random.default_rng(seed)
    curve_types = ["line", "circle", "spiral"]
    # Мало прямых (кривизна=0, не ограничивают V*) — больше кривых с реальными поворотами
    type_weights = np.array([0.0, 0.45, 0.55])   # lines исключены: kappa=0 не ограничивает V*, oracle всегда проваливается

    records: list[dict] = []
    stats = {
        "attempted": 0,
        "invalid_curve": 0,
        "points_total": 0,
        "points_unstable": 0,
        "points_saved": 0,
    }

    t0_total = time.monotonic()
    log.info("Dataset generation started: %d curves requested", num_curves)
    log.info(
        "Drone limits: min_speed=%.2f  max_speed=%.2f  lateral_e_limit=%.2f",
        drone.min_speed, drone.max_speed, drone.lateral_error_limit,
    )
    section_len = (s_end - s_start) / max(n_samples_per_curve - 1, 1)
    log.info(
        "Oracle: horizon=%d (%.2fs)  step=%.2f  section_len=%.2f  coarse_to_fine=%s",
        oracle_cfg.rollout_horizon,
        oracle_cfg.rollout_horizon * ORACLE_DT,
        oracle_cfg.speed_step,
        section_len,
        coarse_to_fine,
    )

    for curve_idx in range(num_curves):
        stats["attempted"] += 1
        t0_curve = time.monotonic()

        # Curve generation.
        curve_type = str(rng.choice(curve_types, p=type_weights))
        spec: CurveSpec = generate_curve(curve_type, rng=rng)

        # Curve validation.
        if not validate_curve(spec.curve.p, s_range=(s_start, s_end)):
            stats["invalid_curve"] += 1
            log.warning(
                "[%d/%d] Curve #%d (%s) failed validate_curve -- skipping",
                curve_idx + 1, num_curves, curve_idx, curve_type,
            )
            continue

        t_norm_val = float(spec.tangent_norm)

        # Sample points.
        s_values = np.linspace(s_start, s_end, n_samples_per_curve)
        curve_records: list[dict] = []

        for s in s_values:
            s = float(s)
            stats["points_total"] += 1

            # Initial state with small perturbations so dynamic features
            # (e1, e2, v_norm, …) are non-zero in the training data.
            state = _make_state_on_curve(spec, s, rng=rng)

            # Features.
            feats = extract_features(state, spec.curve, drone=drone, s=s)

            # Target speed. ``zeta0`` is aligned with the initial point on the curve.
            # gamma_nearest берётся из CurveSpec: γ = 0.2/(||t||²·dt).
            # Это обеспечивает правильный трекинг ближайшей точки при oracle-ролауте.
            V_opt = find_optimal_speed(
                state, spec.curve,
                drone=drone,
                oracle_cfg=oracle_cfg,
                coarse_to_fine=coarse_to_fine,
                dt=oracle_dt,
                kappa=oracle_kappa,
                zeta0=s,
                gamma_nearest=spec.gamma_nearest,
            )

            # Clamp the target to the configured speed range.
            V_opt = float(np.clip(V_opt, drone.min_speed, drone.max_speed))

            # Stability check.
            m = rollout_with_speed(
                state, spec.curve, V_opt,
                oracle_cfg.rollout_horizon,
                drone=drone,
                dt=oracle_dt,
                kappa=oracle_kappa,
                zeta0=s,
                gamma_nearest=spec.gamma_nearest,
            )
            if not is_stable(m, drone):
                stats["points_unstable"] += 1
                log.debug(
                    "  s=%.2f  V_opt=%.2f  unstable (max_e2=%.4f) -- skip",
                    s, V_opt, m["max_e2"],
                )
                continue

            # Record assembly.
            record = _build_record(feats, s, t_norm_val, V_opt)
            curve_records.append(record)
            stats["points_saved"] += 1

        records.extend(curve_records)

        elapsed = time.monotonic() - t0_curve
        log.info(
            "[%d/%d] %s  t_norm=%.3f  saved=%d/%d  V_opt_range=[%.2f, %.2f]  %.1fs",
            curve_idx + 1,
            num_curves,
            curve_type.ljust(6),
            t_norm_val,
            len(curve_records),
            n_samples_per_curve,
            min((r["V_opt"] for r in curve_records), default=float("nan")),
            max((r["V_opt"] for r in curve_records), default=float("nan")),
            elapsed,
        )

    # CSV output.
    _save_csv(records, out_path)

    total_elapsed = time.monotonic() - t0_total
    log.info("-" * 60)
    log.info("Done in %.1fs", total_elapsed)
    log.info("  Curves attempted : %d", stats["attempted"])
    log.info("  Invalid curves   : %d", stats["invalid_curve"])
    log.info("  Points total     : %d", stats["points_total"])
    log.info("  Points unstable  : %d", stats["points_unstable"])
    log.info("  Points saved     : %d", stats["points_saved"])
    log.info("  CSV saved to     : %s", out_path)

    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate V* dataset for MLP training"
    )
    parser.add_argument(
        "--num-curves", type=int, default=10,
        help="Number of curves to generate (default: 10)",
    )
    parser.add_argument(
        "--out", type=str, default=_DEFAULT_OUT_PATH,
        help=f"Output CSV path (default: {_DEFAULT_OUT_PATH})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Samples per curve (default: 5)",
    )
    parser.add_argument(
        "--coarse-fine", action="store_true",
        help="Use coarse-to-fine speed search (default: simple step)",
    )
    args = parser.parse_args()

    generate_dataset(
        num_curves=args.num_curves,
        out_path=args.out,
        seed=args.seed,
        n_samples_per_curve=args.samples,
        coarse_to_fine=args.coarse_fine,
    )
