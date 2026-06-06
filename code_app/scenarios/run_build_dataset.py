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
import pandas as pd
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
    v_opt = data["V_opt"]
    axes[0].hist(v_opt, bins=np.linspace(v_opt.min() - 0.25, v_opt.max() + 0.25, 16), color="steelblue", edgecolor="white")
    # axes[0].violinplot(data["V_opt"])
    axes[0].set_xlabel("$V_{\\mathrm{opt}}$, м/с")
    axes[0].set_ylabel("Количество")
    axes[0].set_title(f"Распределение целевой скорости")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].hist(data["kappa"], bins=30, color="coral", edgecolor="white")
    axes[1].set_xlabel("$\\kappa$ (нормированная кривизна)")
    axes[1].set_ylabel("Количество")
    axes[1].set_title("Распределение кривизны")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    p = os.path.join(out_dir, "dataset_vopt_kappa.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  График: {display_path(p)}")

    # Feature-to-target aggregated plots.
    feature_cols = ["e1", "e2", "de2_dt", "v_norm",
                    "heading_error", "kappa", "kappa_max_lookahead"]
    present = [c for c in feature_cols if c in data]

    n_bins = 8
    cols_per_row = 4
    n_f = len(present)
    nrows = (n_f + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(nrows, cols_per_row, figsize=(cols_per_row * 4, nrows * 3))
    axes = np.array(axes).flatten()

    for i, col in enumerate(present):
        ax = axes[i]

        x = np.asarray(data[col])
        y = np.asarray(data["V_opt"])

        bins = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)
        bin_ids = np.digitize(x, bins) - 1

        centers = []
        means = []
        stds = []
        counts = []

        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue

            centers.append((bins[b] + bins[b + 1]) / 2)
            means.append(np.nanmean(y[mask]))
            stds.append(np.nanstd(y[mask]))
            counts.append(mask.sum())

        ax.plot(
            centers,
            means,
            "o-",
            linewidth=2,
            markersize=4
        )
        # ax.set_ylim(data["V_opt"].min() - 0.2, data["V_opt"].max() + 0.2)

        ax.set_xlabel(f"${col}$" if col in {"e1", "e2", "kappa"} else col)
        ax.set_ylabel("$V_{\\mathrm{opt}}$, м/с")
        ax.set_title(f"{col}: среднее $V_{{\\mathrm{{opt}}}}$ по интервалам")
        ax.grid(True, linestyle="--", alpha=0.5)

    for j in range(n_f, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Зависимость целевой скорости от признаков датасета", fontsize=13)
    fig.tight_layout()
    p = os.path.join(out_dir, "dataset_features_scatter.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  График: {display_path(p)}")

    # Spearman correlation matrix.
    corr_cols = present + ["V_opt"]

    corr_df = pd.DataFrame({
        c: np.asarray(data[c])
        for c in corr_cols
    })

    corr_matrix = corr_df.corr(method="spearman").values

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_masked = np.ma.array(corr_matrix, mask=mask)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        corr_masked,
        vmin=-1,
        vmax=1,
        cmap="coolwarm"
    )

    ax.set_xticks(np.arange(len(corr_cols)))
    ax.set_yticks(np.arange(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticklabels(corr_cols)

    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            if not mask[i, j]:
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=8)

    ax.set_title("Матрица корреляций Спирмена")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    p = os.path.join(out_dir, "dataset_correlations.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", pad_inches=0.05)
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
