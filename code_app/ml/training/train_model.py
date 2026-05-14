"""Training entry point for ``SpeedMLP``."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from drone_sim.models.quad_model import QuadModel
from ml.models.speed_model import INPUT_SIZE, SpeedMLP, save_speed_model

# Logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# CSV columns used for training.
_FEATURE_COLS = [
    "e1", "e2", "de2_dt", "v_norm", "heading_error",
    "kappa", "kappa_max_lookahead",
]
_TARGET_COL = "V_opt"

# Default paths.
_DEFAULT_CSV = "code_app/ml/data/dataset.csv"
_DEFAULT_MODEL = "code_app/ml/data/saved_models/speed_model.pt"


@dataclass
class TrainResult:
    """Training summary."""

    best_val_loss: float
    train_losses: list[float]
    val_losses: list[float]
    stopped_epoch: int
    model_path: str


def load_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load training arrays ``(X, y)`` from CSV."""
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


class _EarlyStopping:
    """Track the best validation loss and stop after ``patience`` misses."""

    def __init__(self, patience: int, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Update state and return ``True`` when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Store the weights of the best epoch.
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Restore the best stored state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train(
    csv_path: str = _DEFAULT_CSV,
    model_path: str = _DEFAULT_MODEL,
    max_speed: float = 10.0,
    n_epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    patience: int = 20,
    seed: int = 42,
    device: str = "cpu",
    drone: Optional[QuadModel] = None,
) -> TrainResult:
    """Train ``SpeedMLP`` and save the best checkpoint.

    Параметры:
        drone — QuadModel, параметры которого использовались при сборке датасета.
                Сохраняются в чекпоинт → SpeedPredictor.load() восстанавливает
                дрона автоматически. None → используются умолчания QuadModel().
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Нормализация drone: если не передан — использовать умолчания
    if drone is None:
        drone = QuadModel()
    log.info(
        "Drone params: min_speed=%.2f  max_speed=%.2f  "
        "lateral_e_limit=%.2f  tang_e_limit=%.2f  max_vel_norm=%.2f",
        drone.min_speed, drone.max_speed,
        drone.lateral_error_limit, drone.tangential_error_limit,
        drone.max_velocity_norm,
    )

    # Dataset.
    log.info("Loading dataset: %s", csv_path)
    X, y = load_dataset(csv_path)
    N = len(X)
    log.info("  Samples: %d  Features: %d", N, X.shape[1])

    if N < 4:
        raise ValueError(f"Dataset too small for train/val split: {N} samples")

    # Train/validation split.
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    n_val = max(1, int(N * val_frac))
    n_train = N - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    log.info("  Train: %d  Val: %d", n_train, n_val)

    # Model, optimizer, loss.
    dev = torch.device(device)
    model = SpeedMLP(max_speed=max_speed, input_size=INPUT_SIZE).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stop = _EarlyStopping(patience=patience)

    log.info("Model: %s", model)
    log.info(
        "Training: epochs=%d  batch=%d  lr=%.0e  patience=%d  device=%s",
        n_epochs, batch_size, lr, patience, device,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    stopped_epoch = n_epochs
    t0 = time.monotonic()

    # Training loop.
    for epoch in range(1, n_epochs + 1):
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

        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            elapsed = time.monotonic() - t0
            log.info(
                "Epoch %4d/%d  train_loss=%.6f  val_loss=%.6f  (%.1fs)",
                epoch, n_epochs, train_loss, val_loss, elapsed,
            )

        if early_stop.step(val_loss, model):
            stopped_epoch = epoch
            log.info(
                "Early stopping at epoch %d  (best val_loss=%.6f)",
                epoch, early_stop.best_loss,
            )
            break

    # Save best checkpoint (with drone_params for reproducible inference).
    early_stop.restore_best(model)
    save_speed_model(model, model_path, drone=drone)

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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SpeedMLP on V* dataset")
    parser.add_argument("--csv", default=_DEFAULT_CSV, help="Path to dataset CSV")
    parser.add_argument("--out", default=_DEFAULT_MODEL, help="Output .pt path")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--max-speed", type=float, default=10.0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
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
