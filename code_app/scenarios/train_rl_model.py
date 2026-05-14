"""Сценарий обучения модели предсказания V* (mlp | sac | td3 | ppo).

Использование (из корня проекта)::

    # Сборка датасета (если ещё нет):
    python code_app/scenarios/run_build_dataset.py --curves 50 --samples 10 --oracle-horizon 4000

    # Обучение конкретной модели:
    python code_app/scenarios/train_rl_model.py --model sac
    python code_app/scenarios/train_rl_model.py --model td3 --epochs 400 --patience 40
    python code_app/scenarios/train_rl_model.py --model ppo --out code_app/ml/data/saved_models/ppo_v1.pt
    python code_app/scenarios/train_rl_model.py --model mlp --csv code_app/ml/data/dataset.csv

    # Запуск сравнения после обучения:
    python code_app/scenarios/run_compare_models.py --models sac,td3,ppo --curve spiral
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

from drone_sim.models.quad_model import QuadModel
from drone_sim.visualization.plotting import display_path
from ml.models.registry import SpeedPredictorAny
from ml.training.train_rl_models import train_rl
from ml.training.train_model import load_dataset, TrainResult

# ---------------------------------------------------------------------------
# Пути по умолчанию
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(__file__)
_DEFAULT_CSV = os.path.join(_HERE, "..", "ml", "data", "dataset.csv")
_DEFAULT_MODELS_DIR = os.path.join(_HERE, "..", "ml", "data", "saved_models")
_DEFAULT_PLOTS = os.path.join(_HERE, "..", "out_images", "training_rl")


def _default_out(model_name: str) -> str:
    return os.path.join(_DEFAULT_MODELS_DIR, f"{model_name}_model.pt")


# ---------------------------------------------------------------------------
# Вывод метрик
# ---------------------------------------------------------------------------

def print_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    result: TrainResult,
) -> None:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    p01 = float(np.mean(np.abs(err) < 0.1) * 100)
    p02 = float(np.mean(np.abs(err) < 0.2) * 100)

    print(f"\n{'='*58}")
    print(f"  МОДЕЛЬ: {model_name.upper()}")
    print(f"{'='*58}")
    print(f"  Best val MSE   : {result.best_val_loss:.6f}")
    print(f"  Test MSE       : {mse:.6f}")
    print(f"  Test RMSE      : {rmse:.6f}")
    print(f"  Test MAE       : {mae:.6f}")
    print(f"  R²             : {r2:.4f}")
    print(f"  |err| < 0.1    : {p01:.1f}%")
    print(f"  |err| < 0.2    : {p02:.1f}%")
    print(f"  Stopped epoch  : {result.stopped_epoch}")
    print(f"  V_pred min/max : {y_pred.min():.3f} / {y_pred.max():.3f}")
    print(f"  V_true min/max : {y_true.min():.3f} / {y_true.max():.3f}")
    print(f"  Model          : {result.model_path}")
    print(f"{'='*58}\n")


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

def plot_training(result: TrainResult, model_name: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(result.train_losses) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, yscale in zip(axes, ["linear", "log"]):
        ax.plot(epochs, result.train_losses, label="Train", color="steelblue", linewidth=1.8)
        ax.plot(epochs, result.val_losses, label="Val", color="coral", linewidth=1.8)
        ax.axvline(result.stopped_epoch, color="gray", linestyle="--", linewidth=1.2,
                   label=f"Early stop (ep {result.stopped_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale(yscale)
        ax.set_title(f"{model_name.upper()} — loss ({'log' if yscale == 'log' else 'linear'})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6, which="both")

    fig.suptitle(
        f"Обучение {model_name.upper()} | best val = {result.best_val_loss:.6f}",
        fontsize=12,
    )
    fig.tight_layout()
    p = os.path.join(out_dir, f"{model_name}_loss.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Loss curves: {display_path(p)}")


def plot_quality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_dir: str,
) -> None:
    err = y_pred - y_true

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ax = axes[0]
    lo = min(y_true.min(), y_pred.min()) - 0.05
    hi = max(y_true.max(), y_pred.max()) + 0.05
    ax.scatter(y_true, y_pred, s=5, alpha=0.35, color="steelblue")
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Идеал")
    ax.set_xlabel("V_opt истинное")
    ax.set_ylabel("V_opt предсказанное")
    ax.set_title(f"{model_name.upper()} — предсказание vs цель")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal")

    ax = axes[1]
    ax.hist(err, bins=40, color="coral", edgecolor="white")
    mae = float(np.mean(np.abs(err)))
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.axvline(mae, color="red", linestyle=":", linewidth=1.5, label=f"MAE={mae:.4f}")
    ax.axvline(-mae, color="red", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Ошибка предсказания")
    ax.set_ylabel("Количество")
    ax.set_title("Распределение ошибки")
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
    centers_arr = np.array(centers)
    w = (edges[1] - edges[0]) * 0.7
    ax.bar(centers_arr, stds, width=w, color="steelblue", alpha=0.7, label="std ошибки")
    ax.plot(centers_arr, means, "ro-", markersize=5, label="mean ошибки")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Диапазон V_opt")
    ax.set_ylabel("Ошибка предсказания")
    ax.set_title("Ошибка по диапазонам")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    p = os.path.join(out_dir, f"{model_name}_quality.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Quality plot : {display_path(p)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обучение модели предсказания V* (mlp | sac | td3 | ppo)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="sac",
        choices=["mlp", "sac", "td3", "ppo"],
        help="Кодовое имя модели",
    )
    parser.add_argument("--csv",       default=_DEFAULT_CSV,   help="Путь к датасету CSV")
    parser.add_argument("--out",       default=None,
                        help="Путь к выходному .pt файлу (по умолчанию: "
                             "code_app/ml/data/saved_models/<model>_model.pt)")
    parser.add_argument("--epochs",    type=int,   default=300,  help="Число эпох")
    parser.add_argument("--batch",     type=int,   default=64,   help="Размер батча")
    parser.add_argument("--lr",        type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience",  type=int,   default=30,   help="Early stopping patience")
    parser.add_argument("--val-frac",  type=float, default=0.2,  help="Доля валидационных данных")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--device",    default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--plots-dir", default=_DEFAULT_PLOTS,  help="Директория для графиков")
    parser.add_argument("--no-plots",  action="store_true",      help="Не строить графики")
    # Drone params
    parser.add_argument("--max-speed",              type=float, default=10.0)
    parser.add_argument("--min-speed",              type=float, default=0.3)
    parser.add_argument("--lateral-error-limit",    type=float, default=0.5)
    parser.add_argument("--tangential-error-limit", type=float, default=0.7)
    parser.add_argument("--max-velocity-norm",      type=float, default=6.0)
    # SAC-specific
    parser.add_argument("--alpha-entropy", type=float, default=0.2,
                        help="[SAC] Вес энтропийного бонуса")
    # TD3-specific
    parser.add_argument("--lambda-q",   type=float, default=0.1,
                        help="[TD3] Вес Q-ведомого обновления актора")
    parser.add_argument("--tau",        type=float, default=0.005,
                        help="[TD3] Polyak tau для целевых сетей")
    parser.add_argument("--actor-delay",type=int,   default=2,
                        help="[TD3] Обновлять актор каждые N батчей")
    # PPO-specific
    parser.add_argument("--ppo-eps",         type=float, default=0.2,
                        help="[PPO] Clip epsilon")
    parser.add_argument("--ppo-mini-epochs", type=int,   default=4,
                        help="[PPO] Мини-эпохи на один проход данных")
    parser.add_argument("--c-value",         type=float, default=0.5,
                        help="[PPO] Коэффициент value loss")
    parser.add_argument("--c-entropy",       type=float, default=0.01,
                        help="[PPO] Коэффициент энтропийного бонуса")
    args = parser.parse_args()

    model_out = args.out or _default_out(args.model)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    drone = QuadModel(
        max_speed=args.max_speed,
        min_speed=args.min_speed,
        lateral_error_limit=args.lateral_error_limit,
        tangential_error_limit=args.tangential_error_limit,
        max_velocity_norm=args.max_velocity_norm,
    )

    print(f"\n{'='*58}")
    print(f"  Обучение модели : {args.model.upper()}")
    print(f"  CSV             : {args.csv}")
    print(f"  Выход           : {model_out}")
    print(f"  Эпохи           : {args.epochs}  patience={args.patience}")
    print(f"  LR              : {args.lr}  batch={args.batch}")
    print(f"  Дрон            : max_speed={drone.max_speed}  min_speed={drone.min_speed}")
    print(f"{'='*58}\n")

    extra = dict(
        alpha_entropy=args.alpha_entropy,
        lambda_q=args.lambda_q,
        tau=args.tau,
        actor_delay=args.actor_delay,
        ppo_eps=args.ppo_eps,
        ppo_mini_epochs=args.ppo_mini_epochs,
        c_value=args.c_value,
        c_entropy=args.c_entropy,
    )

    result = train_rl(
        model_name=args.model,
        csv_path=args.csv,
        model_path=model_out,
        max_speed=args.max_speed,
        n_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        val_frac=args.val_frac,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        drone=drone,
        **extra,
    )

    # Метрики на полном датасете.
    import torch
    X, y = load_dataset(args.csv)
    predictor = SpeedPredictorAny.load(model_out)
    y_pred = predictor._model(torch.from_numpy(X).float()).detach().numpy().flatten()
    y_true = y.flatten()

    print_metrics(args.model, y_true, y_pred, result)

    if not args.no_plots:
        os.makedirs(args.plots_dir, exist_ok=True)
        print("Строю графики...")
        plot_training(result, args.model, args.plots_dir)
        plot_quality(y_true, y_pred, args.model, args.plots_dir)
        print(f"\nГрафики сохранены в: {display_path(args.plots_dir)}")

    print(f"\nМодель '{args.model.upper()}' готова: {display_path(model_out)}")
    print(f"Для сравнения запустите:")
    print(f"  python code_app/scenarios/run_compare_models.py "
          f"--models {args.model} --curve spiral")


if __name__ == "__main__":
    main()
