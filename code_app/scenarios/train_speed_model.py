"""Training scenario for ``SpeedMLP``."""
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

from drone_sim.models.quad_model import QuadModel
from drone_sim.visualization.plotting import display_path
from ml.training.train_model import train, load_dataset, TrainResult
from ml.inference.predict import SpeedPredictor

# Default paths.
_HERE = os.path.dirname(__file__)
_DEFAULT_CSV = os.path.join(_HERE, "..", "ml", "data", "dataset.csv")
_DEFAULT_MODEL = os.path.join(_HERE, "..", "ml", "data", "saved_models", "speed_model.pt")
_DEFAULT_PLOTS = os.path.join(_HERE, "..", "out_images", "training")


def plot_loss_curves(result: TrainResult, out_dir: str) -> None:
    """Plot train and validation losses."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    epochs = list(range(1, len(result.train_losses) + 1))

    ax = axes[0]
    ax.plot(epochs, result.train_losses, label="Train MSE", color="steelblue", linewidth=1.8)
    ax.plot(epochs, result.val_losses, label="Val MSE", color="coral", linewidth=1.8)
    ax.axvline(result.stopped_epoch, color="gray", linestyle="--", linewidth=1.2,
               label=f"Early stop (epoch {result.stopped_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Loss curves")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    ax = axes[1]
    ax.semilogy(epochs, result.train_losses, label="Train MSE", color="steelblue", linewidth=1.8)
    ax.semilogy(epochs, result.val_losses, label="Val MSE", color="coral", linewidth=1.8)
    ax.axvline(result.stopped_epoch, color="gray", linestyle="--", linewidth=1.2,
               label=f"Early stop (epoch {result.stopped_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log)")
    ax.set_title("Loss curves (log scale)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6, which="both")

    fig.suptitle(f"Training SpeedMLP | best val MSE = {result.best_val_loss:.6f}", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, "training_loss.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Потери (loss): {display_path(p)}")


def plot_prediction_quality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,
) -> None:
    """Plot prediction quality diagnostics."""
    err = y_pred - y_true

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ax = axes[0]
    lim_lo = min(y_true.min(), y_pred.min()) - 0.05
    lim_hi = max(y_true.max(), y_pred.max()) + 0.05
    ax.scatter(y_true, y_pred, s=5, alpha=0.35, color="steelblue")
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", linewidth=1.5, label="Ideal")
    ax.set_xlabel("V_opt true")
    ax.set_ylabel("V_opt predicted")
    ax.set_title("Prediction vs target")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")

    ax = axes[1]
    ax.hist(err, bins=40, color="coral", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    mae = float(np.mean(np.abs(err)))
    ax.axvline(mae, color="red", linestyle=":", linewidth=1.5, label=f"MAE={mae:.4f}")
    ax.axvline(-mae, color="red", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Count")
    ax.set_title("Prediction error distribution")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    ax = axes[2]
    n_bins = 8
    edges = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
    centers, means, stds = [], [], []
    for i in range(n_bins):
        mask = (y_true >= edges[i]) & (y_true < edges[i + 1])
        if mask.sum() > 0:
            centers.append((edges[i] + edges[i + 1]) / 2)
            means.append(float(np.mean(err[mask])))
            stds.append(float(np.std(err[mask])))
    centers = np.array(centers)
    means = np.array(means)
    stds = np.array(stds)
    ax.bar(centers, stds, width=(edges[1] - edges[0]) * 0.7,
           color="steelblue", alpha=0.7, label="error std")
    ax.plot(centers, means, "ro-", markersize=5, label="error mean")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("V_opt range")
    ax.set_ylabel("Prediction error")
    ax.set_title("Error by target range")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    p = os.path.join(out_dir, "prediction_quality.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Качество предсказания: {display_path(p)}")


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, result: TrainResult) -> None:
    """Print summary metrics."""
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    pct_within_01 = float(np.mean(np.abs(err) < 0.1) * 100)
    pct_within_02 = float(np.mean(np.abs(err) < 0.2) * 100)

    print(f"\n{'='*55}")
    print("  MODEL METRICS")
    print(f"{'='*55}")
    print(f"  Best val MSE   : {result.best_val_loss:.6f}")
    print(f"  Test MSE       : {mse:.6f}")
    print(f"  Test RMSE      : {rmse:.6f}")
    print(f"  Test MAE       : {mae:.6f}")
    print(f"  R2             : {r2:.4f}")
    print(f"  |err| < 0.1    : {pct_within_01:.1f}%")
    print(f"  |err| < 0.2    : {pct_within_02:.1f}%")
    print(f"  Stopped epoch  : {result.stopped_epoch}")
    print(f"  V_pred min/max : {y_pred.min():.3f} / {y_pred.max():.3f}")
    print(f"  V_true min/max : {y_true.min():.3f} / {y_true.max():.3f}")
    print(f"  Model          : {result.model_path}")
    print(f"{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SpeedMLP and visualize results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",      default=_DEFAULT_CSV,   help="Dataset CSV path")
    parser.add_argument("--out",      default=_DEFAULT_MODEL, help="Output model .pt path")
    parser.add_argument("--epochs",   type=int,   default=200,  help="Max epoch count")
    parser.add_argument("--batch",    type=int,   default=64,   help="Batch size")
    parser.add_argument("--lr",       type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--patience", type=int,   default=20,   help="Early stopping patience")
    parser.add_argument("--val-frac", type=float, default=0.2,  help="Validation fraction")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--plots-dir",default=_DEFAULT_PLOTS,   help="Plot output directory")
    parser.add_argument("--no-plots", action="store_true",       help="Skip plot generation")
    # --- Drone params: must match the values used in run_build_dataset.py ---
    parser.add_argument("--max-speed",              type=float, default=10.0,
                        help="drone.max_speed  — must match dataset generation")
    parser.add_argument("--min-speed",              type=float, default=0.3,
                        help="drone.min_speed  — must match dataset generation")
    parser.add_argument("--lateral-error-limit",    type=float, default=0.5,
                        help="drone.lateral_error_limit  — must match dataset generation")
    parser.add_argument("--tangential-error-limit", type=float, default=0.7,
                        help="drone.tangential_error_limit  — must match dataset generation")
    parser.add_argument("--max-velocity-norm",      type=float, default=6.0,
                        help="drone.max_velocity_norm  — must match dataset generation")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    # Build a single QuadModel — must match the one used in run_build_dataset.py.
    # These parameters are saved into the .pt checkpoint and restored on load.
    drone = QuadModel(
        max_speed=args.max_speed,
        min_speed=args.min_speed,
        lateral_error_limit=args.lateral_error_limit,
        tangential_error_limit=args.tangential_error_limit,
        max_velocity_norm=args.max_velocity_norm,
    )

    print("\nTraining SpeedMLP")
    print(f"  CSV      : {args.csv}")
    print(f"  Model    : {args.out}")
    print(f"  Epochs   : {args.epochs}  patience={args.patience}")
    print(f"  LR       : {args.lr}  batch={args.batch}")
    print(f"  Drone    : max_speed={drone.max_speed}  min_speed={drone.min_speed}")
    print(f"             lateral_e_lim={drone.lateral_error_limit}  "
          f"tang_e_lim={drone.tangential_error_limit}  "
          f"max_vel_norm={drone.max_velocity_norm}\n")

    result = train(
        csv_path=args.csv,
        model_path=args.out,
        max_speed=args.max_speed,
        n_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        val_frac=args.val_frac,
        patience=args.patience,
        seed=args.seed,
        drone=drone,
    )

    X, y = load_dataset(args.csv)
    predictor = SpeedPredictor.load(args.out)
    import torch
    y_pred = predictor._model(torch.from_numpy(X).float()).detach().numpy().flatten()
    y_true = y.flatten()

    print_metrics(y_true, y_pred, result)

    if not args.no_plots:
        print("Строю графики качества обучения...")
        plot_loss_curves(result, args.plots_dir)
        plot_prediction_quality(y_true, y_pred, args.plots_dir)
        print(f"\nГрафики сохранены в: {display_path(args.plots_dir)}")


if __name__ == "__main__":
    main()
