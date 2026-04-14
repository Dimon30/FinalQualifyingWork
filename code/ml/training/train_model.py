"""
ml/training/train_model.py
============================
Обучение SpeedMLP на датасете из CSV.

Пайплайн:
    1. Загрузить CSV (признаки + V_opt)
    2. Train / val split (80/20 по умолчанию)
    3. Обучить SpeedMLP: loss = MSE(V_pred, V_opt)
    4. Early stopping по val_loss
    5. Сохранить лучшую модель в .pt

Признаки (входы модели, 7 колонок):
    e1, e2, de2_dt, v_norm, heading_error, kappa, kappa_max_lookahead

Целевая переменная:
    V_opt

Колонки s, t_norm в CSV игнорируются при обучении (диагностика).

Публичный API:
    train(csv_path, model_path, ...) -> TrainResult
    load_dataset(csv_path)           -> (X, y)

CLI:
    python -m ml.training.train_model --csv code/ml/data/dataset.csv
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from ml.models.speed_model import INPUT_SIZE, SpeedMLP, save_speed_model

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Имена признаков в CSV (строго в этом порядке)
_FEATURE_COLS = [
    "e1", "e2", "de2_dt", "v_norm", "heading_error",
    "kappa", "kappa_max_lookahead",
]
_TARGET_COL = "V_opt"

# Пути по умолчанию
_DEFAULT_CSV   = "code/ml/data/dataset.csv"
_DEFAULT_MODEL = "code/ml/data/model.pt"


# ---------------------------------------------------------------------------
# Результат обучения
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Итоги обучения.

    Атрибуты:
        best_val_loss  — лучший val MSE (по которому сохранена модель)
        train_losses   — список train MSE по эпохам
        val_losses     — список val MSE по эпохам
        stopped_epoch  — эпоха, на которой сработал early stopping (или n_epochs)
        model_path     — путь к сохранённой модели
    """
    best_val_loss: float
    train_losses:  list[float]
    val_losses:    list[float]
    stopped_epoch: int
    model_path:    str


# ---------------------------------------------------------------------------
# Загрузка датасета
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Загрузить CSV и вернуть (X, y).

    Параметры:
        csv_path — путь к CSV, сгенерированному build_dataset.py

    Возвращает:
        X : ndarray shape (N, 7)  — признаки float32
        y : ndarray shape (N, 1)  — V_opt float32

    Исключения:
        FileNotFoundError — файл не найден
        ValueError        — отсутствуют нужные колонки
    """
    import csv

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    missing = [c for c in _FEATURE_COLS + [_TARGET_COL] if c not in rows[0]]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = np.array(
        [[float(r[c]) for c in _FEATURE_COLS] for r in rows],
        dtype=np.float32,
    )
    y = np.array(
        [[float(r[_TARGET_COL])] for r in rows],
        dtype=np.float32,
    )
    return X, y


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class _EarlyStopping:
    """Останавливает обучение если val_loss не улучшается patience эпох подряд."""

    def __init__(self, patience: int, min_delta: float = 1e-6) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Обновить состояние. Возвращает True если нужно остановиться."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            # Сохраняем копию весов лучшей эпохи
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Восстановить веса лучшей эпохи в модель."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Основная функция обучения
# ---------------------------------------------------------------------------

def train(
    csv_path:    str   = _DEFAULT_CSV,
    model_path:  str   = _DEFAULT_MODEL,
    max_speed:   float = 3.0,
    n_epochs:    int   = 200,
    batch_size:  int   = 64,
    lr:          float = 1e-3,
    val_frac:    float = 0.2,
    patience:    int   = 20,
    seed:        int   = 42,
    device:      str   = "cpu",
) -> TrainResult:
    """Обучить SpeedMLP и сохранить лучшую модель.

    Параметры:
        csv_path   — путь к CSV с датасетом
        model_path — куда сохранить .pt файл
        max_speed  — верхняя граница выхода модели (должна совпадать с drone.max_speed)
        n_epochs   — максимальное число эпох
        batch_size — размер мини-батча
        lr         — learning rate (Adam)
        val_frac   — доля val-выборки (0.2 = 20%)
        patience   — число эпох без улучшения val_loss до остановки
        seed       — seed для воспроизводимости split
        device     — 'cpu' или 'cuda'

    Возвращает:
        TrainResult с историей loss и путём к модели
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 1. Загрузка данных ---
    log.info("Loading dataset: %s", csv_path)
    X, y = load_dataset(csv_path)
    N = len(X)
    log.info("  Samples: %d  Features: %d", N, X.shape[1])

    if N < 4:
        raise ValueError(f"Dataset too small for train/val split: {N} samples")

    # --- 2. Train / val split ---
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    n_val   = max(1, int(N * val_frac))
    n_train = N - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    log.info("  Train: %d  Val: %d", n_train, n_val)

    # --- 3. Модель, оптимизатор, loss ---
    dev = torch.device(device)
    model = SpeedMLP(max_speed=max_speed, input_size=INPUT_SIZE).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stop = _EarlyStopping(patience=patience)

    log.info("Model: %s", model)
    log.info("Training: epochs=%d  batch=%d  lr=%.0e  patience=%d  device=%s",
             n_epochs, batch_size, lr, patience, device)

    train_losses: list[float] = []
    val_losses:   list[float] = []
    stopped_epoch = n_epochs
    t0 = time.monotonic()

    # --- 4. Цикл обучения ---
    for epoch in range(1, n_epochs + 1):

        # -- Train --
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
        train_loss = train_loss_sum / n_train

        # -- Val --
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb)
                val_loss_sum += criterion(pred, yb).item() * len(xb)
        val_loss = val_loss_sum / n_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Логируем каждые 10 эпох и последнюю
        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            elapsed = time.monotonic() - t0
            log.info(
                "Epoch %4d/%d  train_loss=%.6f  val_loss=%.6f  (%.1fs)",
                epoch, n_epochs, train_loss, val_loss, elapsed,
            )

        # -- Early stopping --
        if early_stop.step(val_loss, model):
            stopped_epoch = epoch
            log.info(
                "Early stopping at epoch %d  (best val_loss=%.6f)",
                epoch, early_stop.best_loss,
            )
            break

    # --- 5. Восстановить лучшие веса и сохранить ---
    early_stop.restore_best(model)
    save_speed_model(model, model_path)

    elapsed_total = time.monotonic() - t0
    log.info("-" * 60)
    log.info("Training finished in %.1fs", elapsed_total)
    log.info("  Best val_loss : %.6f", early_stop.best_loss)
    log.info("  Stopped epoch : %d / %d", stopped_epoch, n_epochs)
    log.info("  Model saved   : %s", model_path)

    return TrainResult(
        best_val_loss=early_stop.best_loss,
        train_losses=train_losses,
        val_losses=val_losses,
        stopped_epoch=stopped_epoch,
        model_path=model_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SpeedMLP on V* dataset")
    parser.add_argument("--csv",      default=_DEFAULT_CSV,   help="Path to dataset CSV")
    parser.add_argument("--out",      default=_DEFAULT_MODEL, help="Output .pt path")
    parser.add_argument("--epochs",   type=int,   default=200)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--patience", type=int,   default=20)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--max-speed",type=float, default=3.0)
    parser.add_argument("--device",   default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    train(
        csv_path=args.csv,
        model_path=args.out,
        max_speed=args.max_speed,
        n_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        val_frac=args.val_frac,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )
