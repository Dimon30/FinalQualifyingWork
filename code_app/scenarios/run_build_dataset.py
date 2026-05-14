"""Dataset generation scenario."""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_sim.models.quad_model import QuadModel
from drone_sim.visualization.plotting import display_path
from ml.config import ORACLE_DT, ORACLE_KAPPA, OracleConfig
from ml.dataset.build_dataset import generate_dataset

# Output paths.
_HERE = os.path.dirname(__file__)
_DEFAULT_CSV = os.path.join(_HERE, "..", "ml", "data", "dataset.csv")
_DEFAULT_OUT_IMG = os.path.join(_HERE, "..", "out_images", "dataset")


def plot_dataset_stats(csv_path: str, out_dir: str) -> None:
    """Plot basic statistics for the generated CSV dataset."""
    import csv as csvmod

    os.makedirs(out_dir, exist_ok=True)

    rows: list[dict] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    if not rows:
        print("  CSV is empty; plots are skipped.")
        return

    keys = list(rows[0].keys())
    data = {k: np.array([r[k] for r in rows]) for k in keys}
    N = len(rows)

    # Distribution of V_opt and curvature.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(data["V_opt"], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("V_opt")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Target speed distribution (N={N})")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].hist(data["kappa"], bins=30, color="coral", edgecolor="white")
    axes[1].set_xlabel("kappa (normalized)")
    axes[1].set_title("Curvature distribution")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    p = os.path.join(out_dir, "dataset_vopt_kappa.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  График: {display_path(p)}")

    # Feature-to-target scatter plots.
    feature_cols = ["e1", "e2", "de2_dt", "v_norm",
                    "heading_error", "kappa", "kappa_max_lookahead"]
    present = [c for c in feature_cols if c in data]

    n_f = len(present)
    cols_per_row = 4
    nrows = (n_f + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(nrows, cols_per_row, figsize=(cols_per_row * 4, nrows * 3))
    axes = np.array(axes).flatten()

    for i, col in enumerate(present):
        ax = axes[i]
        ax.scatter(data[col], data["V_opt"], s=4, alpha=0.4, color="steelblue")
        try:
            z = np.polyfit(data[col], data["V_opt"], 1)
            xline = np.linspace(data[col].min(), data[col].max(), 50)
            ax.plot(xline, np.polyval(z, xline), "r-", linewidth=1.5)
        except Exception:
            pass
        ax.set_xlabel(col)
        ax.set_ylabel("V_opt")
        ax.set_title(f"{col} vs V_opt")
        ax.grid(True, linestyle="--", alpha=0.5)

    for j in range(n_f, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature scatter plots", fontsize=13)
    fig.tight_layout()
    p = os.path.join(out_dir, "dataset_features_scatter.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  График: {display_path(p)}")

    # Correlation with V_opt.
    corrs = []
    for col in present:
        c = float(np.corrcoef(data[col], data["V_opt"])[0, 1])
        corrs.append((col, c))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    names = [c[0] for c in corrs]
    vals = [c[1] for c in corrs]
    colors = ["steelblue" if v >= 0 else "coral" for v in vals]
    ax.barh(names, vals, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with V_opt")
    ax.set_title("Feature correlation")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    fig.tight_layout()
    p = os.path.join(out_dir, "dataset_correlations.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  График: {display_path(p)}")


def print_csv_summary(csv_path: str) -> None:
    """Print basic CSV statistics."""
    import csv as csvmod

    rows: list[dict] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    if not rows:
        print("  Dataset is empty.")
        return

    N = len(rows)
    V_opt = np.array([r["V_opt"] for r in rows])
    t_norm = np.array([r["t_norm"] for r in rows])

    print(f"\n{'='*55}")
    print("  DATASET SUMMARY")
    print(f"{'='*55}")
    print(f"  Records         : {N}")
    print(f"  V_opt min/mean/max : {V_opt.min():.3f} / {V_opt.mean():.3f} / {V_opt.max():.3f}")
    print(f"  t_norm min/mean/max: {t_norm.min():.3f} / {t_norm.mean():.3f} / {t_norm.max():.3f}")
    unique_tn = np.unique(np.round(t_norm, 2))
    print(f"  Unique ||t||    : {len(unique_tn)}")
    print(f"{'='*55}\n")


def main() -> None:
    _d = QuadModel()    # источник дефолтов дрона — единственная точка правды
    _o = OracleConfig() # источник дефолтов оракла — единственная точка правды

    parser = argparse.ArgumentParser(
        description="Generate a V* dataset for SpeedMLP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Dataset size ---
    parser.add_argument("--curves",  type=int,   default=1000,         help="Number of curves")
    parser.add_argument("--samples", type=int,   default=10,           help="Samples per curve")
    parser.add_argument("--out",     type=str,   default=_DEFAULT_CSV, help="Output CSV path (filename)")
    parser.add_argument("--seed",    type=int,   default=30,           help="Random seed")
    parser.add_argument("--coarse-fine", action="store_true",          help="Coarse-to-fine oracle (slower, more precise)")
    # --- Drone params (define feature normalization scales AND oracle V* search range) ---
    parser.add_argument("--max-speed",              type=float, default=_d.max_speed,              help="drone.max_speed  (V* upper bound, normalises v_norm)")
    parser.add_argument("--min-speed",              type=float, default=_d.min_speed,              help="drone.min_speed  (V* lower bound)")
    parser.add_argument("--lateral-error-limit",    type=float, default=_d.lateral_error_limit,    help="drone.lateral_error_limit  (stability & e2 scale)")
    parser.add_argument("--tangential-error-limit", type=float, default=_d.tangential_error_limit, help="drone.tangential_error_limit  (e1 scale)")
    parser.add_argument("--max-velocity-norm",      type=float, default=_d.max_velocity_norm,      help="drone.max_velocity_norm  (explosion threshold & de2_dt scale)")
    # --- Oracle params ---
    parser.add_argument("--oracle-horizon",      type=int,   default=_o.rollout_horizon, help="oracle.rollout_horizon  (шагов RK4 на один ролаут)")
    parser.add_argument("--oracle-speed-step",   type=float, default=_o.speed_step,     help="oracle.speed_step  (шаг перебора V* в линейном режиме)")
    parser.add_argument("--oracle-coarse-step",  type=float, default=_o.coarse_step,    help="oracle.coarse_step  (грубый шаг при coarse-to-fine)")
    parser.add_argument("--oracle-fine-step",    type=float, default=_o.fine_step,      help="oracle.fine_step  (точный шаг при coarse-to-fine)")
    parser.add_argument("--oracle-min-stable",   type=int,   default=_o.min_stable_steps, help="oracle.min_stable_steps  (минимум стабильных шагов для зачёта)")
    parser.add_argument("--oracle-dt",           type=float, default=ORACLE_DT,           help="Oracle RK4 step dt  (default 0.01; уменьшить при нестабильном ролауте)")
    parser.add_argument("--oracle-kappa",        type=float, default=ORACLE_KAPPA,        help="Oracle observer kappa  (default 100; уменьшить при нестабильном ролауте)")
    # --- Output ---
    parser.add_argument("--no-plots",   action="store_true",              help="Skip diagnostic plots")
    parser.add_argument("--plots-dir",  type=str, default=_DEFAULT_OUT_IMG, help="Plot output directory")
    args = parser.parse_args()

    # Build a single QuadModel — the source of truth for this dataset run.
    # The SAME parameters must be passed to train_speed_model.py and run_nn_speed.py.
    drone = QuadModel(
        max_speed=args.max_speed,
        min_speed=args.min_speed,
        lateral_error_limit=args.lateral_error_limit,
        tangential_error_limit=args.tangential_error_limit,
        max_velocity_norm=args.max_velocity_norm,
    )

    oracle_cfg = OracleConfig(
        rollout_horizon=args.oracle_horizon,
        speed_step=args.oracle_speed_step,
        coarse_step=args.oracle_coarse_step,
        fine_step=args.oracle_fine_step,
        min_stable_steps=args.oracle_min_stable,
    )

    expected = args.curves * args.samples
    print("\nDataset generation")
    print(f"  Curves          : {args.curves}")
    print(f"  Samples/curve   : {args.samples}")
    print(f"  Expected rows   : ~{expected}")
    print(f"  Oracle mode     : {'coarse-to-fine' if args.coarse_fine else 'linear'}")
    print(f"  CSV             : {args.out}")
    print(f"  Drone  max_speed={drone.max_speed}  min_speed={drone.min_speed}")
    print(f"         lateral_e_lim={drone.lateral_error_limit}  "
          f"tang_e_lim={drone.tangential_error_limit}  "
          f"max_vel_norm={drone.max_velocity_norm}")
    print(f"  Oracle horizon={oracle_cfg.rollout_horizon}  dt={args.oracle_dt}  kappa={args.oracle_kappa}")
    print(f"         speed_step={oracle_cfg.speed_step}  coarse_step={oracle_cfg.coarse_step}  "
          f"fine_step={oracle_cfg.fine_step}  min_stable={oracle_cfg.min_stable_steps}\n")

    t0 = time.monotonic()
    out_path = generate_dataset(
        num_curves=args.curves,
        out_path=args.out,
        seed=args.seed,
        n_samples_per_curve=args.samples,
        coarse_to_fine=args.coarse_fine,
        drone=drone,
        oracle_cfg=oracle_cfg,
        oracle_dt=args.oracle_dt,
        oracle_kappa=args.oracle_kappa,
    )
    elapsed = time.monotonic() - t0

    print(f"\nГотово за {elapsed:.1f} с")
    print_csv_summary(out_path)

    if not args.no_plots:
        print("Строю диагностические графики...")
        plot_dataset_stats(out_path, args.plots_dir)
        print(f"Графики сохранены в {display_path(args.plots_dir)}")


if __name__ == "__main__":
    main()
