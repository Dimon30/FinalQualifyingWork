"""
Обучение SpeedMLP на датасете V*.

Что делает:
    1. Загружает CSV-датасет
    2. Обучает SpeedMLP с ранней остановкой
    3. Строит графики обучения:
        - loss curves (train/val MSE по эпохам)
        - scatter: предсказание vs истинное значение V_opt
        - гистограмма ошибок предсказания
        - distribution V_pred по диапазонам
    4. Выводит итоговые метрики (MSE, MAE, R², диапазон предсказаний)
    5. Сохраняет модель в .pt

Запуск (из корня проекта):
    python code/scenarios/train_speed_model.py
    python code/scenarios/train_speed_model.py --csv code/ml/data/dataset.csv --epochs 300
    python code/scenarios/train_speed_model.py --lr 5e-4 --patience 30 --out code/ml/data/model/speed_model.pt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.training.train_model import train, load_dataset, TrainResult
from ml.inference.predict import SpeedPredictor
from ml.dataset.features import feature_vector

# ---------------------------------------------------------------------------
# Пути по умолчанию
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_DEFAULT_CSV   = os.path.join(_HERE, "..", "ml", "data", "dataset.csv")
_DEFAULT_MODEL = os.path.join(_HERE, "..", "ml", "data", "model", "speed_model.pt")
_DEFAULT_PLOTS = os.path.join(_HERE, "..", "out_images", "training")


# ---------------------------------------------------------------------------
# Визуализация обучения
# ---------------------------------------------------------------------------

def plot_loss_curves(result: TrainResult, out_dir: str) -> None:
    """Loss-кривые train/val по эпохам."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    epochs = list(range(1, len(result.train_losses) + 1))

    # --- Линейная шкала ---
    ax = axes[0]
    ax.plot(epochs, result.train_losses, label="Train MSE", color="steelblue", linewidth=1.8)
    ax.plot(epochs, result.val_losses,   label="Val MSE",   color="coral",      linewidth=1.8)
    ax.axvline(result.stopped_epoch, color="gray", linestyle="--",
               linewidth=1.2, label=f"Early stop (epoch {result.stopped_epoch})")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("MSE")
    ax.set_title("Кривые потерь (линейная шкала)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # --- Логарифмическая шкала ---
    ax = axes[1]
    ax.semilogy(epochs, result.train_losses, label="Train MSE", color="steelblue", linewidth=1.8)
    ax.semilogy(epochs, result.val_losses,   label="Val MSE",   color="coral",      linewidth=1.8)
    ax.axvline(result.stopped_epoch, color="gray", linestyle="--",
               linewidth=1.2, label=f"Early stop (epoch {result.stopped_epoch})")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("MSE (log)")
    ax.set_title("Кривые потерь (log-шкала)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6, which="both")

    fig.suptitle(f"Обучение SpeedMLP  |  best val MSE = {result.best_val_loss:.6f}", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, "training_loss.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Loss-кривые: {p}")


def plot_prediction_quality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,
) -> None:
    """Scatter: предсказание vs истинное + гистограмма ошибок."""
    err = y_pred - y_true

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- 1. Scatter ---
    ax = axes[0]
    lim_lo = min(y_true.min(), y_pred.min()) - 0.05
    lim_hi = max(y_true.max(), y_pred.max()) + 0.05
    ax.scatter(y_true, y_pred, s=5, alpha=0.35, color="steelblue")
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", linewidth=1.5, label="Идеал")
    ax.set_xlabel("V_opt истинное")
    ax.set_ylabel("V_opt предсказанное")
    ax.set_title("Предсказание vs Истина")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")

    # --- 2. Гистограмма ошибок ---
    ax = axes[1]
    ax.hist(err, bins=40, color="coral", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    mae = float(np.mean(np.abs(err)))
    ax.axvline(mae, color="red", linestyle=":", linewidth=1.5, label=f"MAE={mae:.4f}")
    ax.axvline(-mae, color="red", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Ошибка (V_pred - V_opt)")
    ax.set_ylabel("Число")
    ax.set_title("Распределение ошибок предсказания")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- 3. Разброс ошибок по диапазонам V_opt ---
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
           color="steelblue", alpha=0.7, label="std ошибки")
    ax.plot(centers, means, "ro-", markersize=5, label="mean ошибки")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("V_opt (диапазон)")
    ax.set_ylabel("Ошибка предсказания")
    ax.set_title("Ошибка по диапазонам V*")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    p = os.path.join(out_dir, "prediction_quality.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Качество предсказания: {p}")


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, result: TrainResult) -> None:
    """Итоговые метрики в терминале."""
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
    print(f"  ИТОГОВЫЕ МЕТРИКИ МОДЕЛИ")
    print(f"{'='*55}")
    print(f"  Best val MSE   : {result.best_val_loss:.6f}")
    print(f"  Test MSE       : {mse:.6f}")
    print(f"  Test RMSE      : {rmse:.6f}")
    print(f"  Test MAE       : {mae:.6f}")
    print(f"  R²             : {r2:.4f}")
    print(f"  |err| < 0.1    : {pct_within_01:.1f}%")
    print(f"  |err| < 0.2    : {pct_within_02:.1f}%")
    print(f"  Остановка      : эпоха {result.stopped_epoch}")
    print(f"  V_pred  min/max: {y_pred.min():.3f} / {y_pred.max():.3f}")
    print(f"  V_true  min/max: {y_true.min():.3f} / {y_true.max():.3f}")
    print(f"  Модель         : {result.model_path}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обучение SpeedMLP и визуализация результатов",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",      default=_DEFAULT_CSV,   help="Путь к CSV-датасету")
    parser.add_argument("--out",      default=_DEFAULT_MODEL, help="Путь для сохранения .pt модели")
    parser.add_argument("--epochs",   type=int,   default=200,  help="Макс. число эпох")
    parser.add_argument("--batch",    type=int,   default=64,   help="Размер мини-батча")
    parser.add_argument("--lr",       type=float, default=1e-3, help="Learning rate (Adam)")
    parser.add_argument("--patience", type=int,   default=20,   help="Эпох без улучшения до остановки")
    parser.add_argument("--val-frac", type=float, default=0.2,  help="Доля val-выборки")
    parser.add_argument("--max-speed",type=float, default=3.0,  help="max_speed дрона (= верх. граница выхода)")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--plots-dir",default=_DEFAULT_PLOTS,   help="Директория для графиков")
    parser.add_argument("--no-plots", action="store_true",       help="Не строить графики")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    print(f"\nОбучение SpeedMLP")
    print(f"  CSV      : {args.csv}")
    print(f"  Модель   : {args.out}")
    print(f"  Эпохи    : {args.epochs}  patience={args.patience}")
    print(f"  LR       : {args.lr}  batch={args.batch}\n")

    # --- 1. Обучение ---
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
    )

    # --- 2. Предсказания на всём датасете ---
    X, y = load_dataset(args.csv)
    predictor = SpeedPredictor.load(args.out)
    import torch
    y_pred = predictor._model(
        torch.from_numpy(X).float()
    ).detach().numpy().flatten()
    y_true = y.flatten()

    print_metrics(y_true, y_pred, result)

    if not args.no_plots:
        print("Строю графики...")
        plot_loss_curves(result, args.plots_dir)
        plot_prediction_quality(y_true, y_pred, args.plots_dir)
        print(f"\nВсе графики сохранены в: {args.plots_dir}")


if __name__ == "__main__":
    main()
