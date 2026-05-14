"""Обучение RL-архитектур (SAC, TD3, PPO) на датасете V*.

Все три алгоритма работают в режиме offline supervised regression:
принимают тот же CSV датасет что и SpeedMLP, но обучаются с разными
функциями потерь и архитектурными особенностями.

SAC:  Gaussian NLL + entropy bonus + twin critics.
TD3:  BC MSE + Q-guided actor update (delayed) + Polyak target networks.
PPO:  Clipped surrogate + value MSE + entropy bonus (offline BC-PPO).

Публичный API::

    from ml.training.train_rl_models import train_rl

    result = train_rl("sac", csv_path="...", model_path="...", n_epochs=200)
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from drone_sim.models.quad_model import QuadModel
from ml.models.registry import get_speed_model, save_speed_model_any, MODEL_NAMES
from ml.models.sac_model import SpeedSAC
from ml.models.td3_model import SpeedTD3
from ml.models.ppo_model import SpeedPPO
from ml.training.train_model import TrainResult, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Общий хелпер early stopping (упрощённый, без зависимости от train_model)
# ---------------------------------------------------------------------------

class _ES:
    def __init__(self, patience: int, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.best_state: Optional[dict] = None

    def step(self, val: float, model: nn.Module) -> bool:
        if val < self.best - self.min_delta:
            self.best = val
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _make_loaders(X, y, val_frac, batch_size, seed):
    N = len(X)
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    n_val = max(1, int(N * val_frac))
    n_train = N - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, n_train, n_val


def _val_mse(model: nn.Module, loader: DataLoader, dev: torch.device) -> float:
    """MSE(mean V*, V_opt) на валидационной выборке (для early stopping)."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            pred = model(xb)
            total += ((pred - yb) ** 2).sum().item()
            n += len(xb)
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# SAC training
# ---------------------------------------------------------------------------

def train_sac(
    csv_path: str,
    model_path: str,
    max_speed: float = 10.0,
    n_epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    patience: int = 30,
    seed: int = 42,
    device: str = "cpu",
    drone: Optional[QuadModel] = None,
    alpha_entropy: float = 0.2,
) -> TrainResult:
    """Обучить SpeedSAC на датасете V*.

    Параметры:
        alpha_entropy — вес энтропийного бонуса (0 = чистый NLL, 1 = MSE без σ).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    if drone is None:
        drone = QuadModel()

    log.info("SAC — загружаю датасет: %s", csv_path)
    X, y = load_dataset(csv_path)
    log.info("  Сэмплов: %d  Признаков: %d", len(X), X.shape[1])
    train_loader, val_loader, n_train, n_val = _make_loaders(X, y, val_frac, batch_size, seed)
    log.info("  Train: %d  Val: %d", n_train, n_val)

    model: SpeedSAC = get_speed_model("sac", max_speed=max_speed).to(dev)
    log.info("Модель: %s", model)

    # Раздельные оптимизаторы: критик обновляется независимо от актора.
    opt_actor = torch.optim.Adam(
        list(model.actor_backbone.parameters())
        + list(model.mean_head.parameters())
        + list(model.log_std_head.parameters()),
        lr=lr,
    )
    opt_critic = torch.optim.Adam(
        list(model.q1.parameters()) + list(model.q2.parameters()),
        lr=lr,
    )
    mse = nn.MSELoss()
    es = _ES(patience)

    train_losses, val_losses = [], []
    stopped = n_epochs
    t0 = time.monotonic()

    for epoch in range(1, n_epochs + 1):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)

            # --- Critic update ---
            opt_critic.zero_grad()
            q1_out, q2_out = model.q_values(xb, yb)
            loss_critic = mse(q1_out, yb) + mse(q2_out, yb)
            loss_critic.backward()
            opt_critic.step()

            # --- Actor update (Gaussian NLL + entropy) ---
            opt_actor.zero_grad()
            mean, std = model.forward_actor(xb)
            # NLL = 0.5 * ((y - mu)/sigma)^2 + log(sigma)
            nll = 0.5 * ((yb - mean) / std) ** 2 + torch.log(std)
            # Entropy = 0.5 * log(2*pi*e*sigma^2) ∝ log(sigma).
            # alpha_entropy > 0 → снижаем штраф за log(sigma) → шире sigma.
            loss_actor = (nll - alpha_entropy * torch.log(std)).mean()
            loss_actor.backward()
            opt_actor.step()

            tl += loss_actor.item() * len(xb)

        train_loss = tl / n_train
        val_loss = _val_mse(model, val_loader, dev)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 20 == 0 or epoch == 1:
            log.info(
                "Epoch %4d/%d  actor_loss=%.5f  val_MSE=%.5f  (%.1fs)",
                epoch, n_epochs, train_loss, val_loss, time.monotonic() - t0,
            )

        if es.step(val_loss, model):
            stopped = epoch
            log.info("Early stop epoch=%d  best_val=%.5f", epoch, es.best)
            break

    es.restore(model)
    save_speed_model_any(model, model_path, drone=drone)
    log.info("Модель сохранена: %s", model_path)

    return TrainResult(
        best_val_loss=es.best,
        train_losses=train_losses,
        val_losses=val_losses,
        stopped_epoch=stopped,
        model_path=model_path,
    )


# ---------------------------------------------------------------------------
# TD3 training
# ---------------------------------------------------------------------------

def train_td3(
    csv_path: str,
    model_path: str,
    max_speed: float = 10.0,
    n_epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    patience: int = 30,
    seed: int = 42,
    device: str = "cpu",
    drone: Optional[QuadModel] = None,
    lambda_q: float = 0.1,
    tau: float = 0.005,
    actor_delay: int = 2,
) -> TrainResult:
    """Обучить SpeedTD3 на датасете V*.

    Параметры:
        lambda_q    — вес Q-ведомого обновления актора (BC + lambda_q * (-Q)).
        tau         — скорость Polyak-усреднения целевых сетей.
        actor_delay — обновлять актор каждые N батчей (TD3 delayed update).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    if drone is None:
        drone = QuadModel()

    log.info("TD3 — загружаю датасет: %s", csv_path)
    X, y = load_dataset(csv_path)
    log.info("  Сэмплов: %d  Признаков: %d", len(X), X.shape[1])
    train_loader, val_loader, n_train, n_val = _make_loaders(X, y, val_frac, batch_size, seed)
    log.info("  Train: %d  Val: %d", n_train, n_val)

    model: SpeedTD3 = get_speed_model("td3", max_speed=max_speed).to(dev)
    log.info("Модель: %s", model)

    actor_params = list(model.actor.parameters())
    critic_params = list(model.q1.parameters()) + list(model.q2.parameters())
    opt_actor = torch.optim.Adam(actor_params, lr=lr)
    opt_critic = torch.optim.Adam(critic_params, lr=lr)
    mse = nn.MSELoss()
    es = _ES(patience)

    train_losses, val_losses = [], []
    stopped = n_epochs
    t0 = time.monotonic()
    batch_counter = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            batch_counter += 1

            # --- Critic update ---
            opt_critic.zero_grad()
            q1_out, q2_out = model.q_values(xb, yb)
            loss_c = mse(q1_out, yb) + mse(q2_out, yb)
            loss_c.backward()
            opt_critic.step()

            # --- Actor update (delayed) ---
            if batch_counter % actor_delay == 0:
                opt_actor.zero_grad()
                pred = model(xb)          # actor prediction
                # Behavioral cloning term.
                loss_bc = mse(pred, yb)
                # Q-guided: maximize min(Q1, Q2)(x, actor(x)).
                # Q-критики обучены предсказывать V_opt → Q(x, pred) ≈ pred
                # при pred → V_opt. Минимизируем -Q → actor движется к V_opt.
                q_guide = model.q_target_min(xb, pred)
                loss_actor = loss_bc - lambda_q * q_guide.mean() / (max_speed + 1e-8)
                loss_actor.backward()
                opt_actor.step()

                # Polyak update.
                model.update_targets(tau=tau)
                tl += loss_bc.item() * len(xb)

        train_loss = tl / max(n_train, 1)
        val_loss = _val_mse(model, val_loader, dev)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 20 == 0 or epoch == 1:
            log.info(
                "Epoch %4d/%d  bc_loss=%.5f  val_MSE=%.5f  (%.1fs)",
                epoch, n_epochs, train_loss, val_loss, time.monotonic() - t0,
            )

        if es.step(val_loss, model):
            stopped = epoch
            log.info("Early stop epoch=%d  best_val=%.5f", epoch, es.best)
            break

    es.restore(model)
    save_speed_model_any(model, model_path, drone=drone)
    log.info("Модель сохранена: %s", model_path)

    return TrainResult(
        best_val_loss=es.best,
        train_losses=train_losses,
        val_losses=val_losses,
        stopped_epoch=stopped,
        model_path=model_path,
    )


# ---------------------------------------------------------------------------
# PPO training (offline BC-PPO)
# ---------------------------------------------------------------------------

def train_ppo(
    csv_path: str,
    model_path: str,
    max_speed: float = 10.0,
    n_epochs: int = 300,
    batch_size: int = 64,
    lr: float = 3e-4,
    val_frac: float = 0.2,
    patience: int = 30,
    seed: int = 42,
    device: str = "cpu",
    drone: Optional[QuadModel] = None,
    ppo_eps: float = 0.2,
    ppo_mini_epochs: int = 4,
    c_value: float = 0.5,
    c_entropy: float = 0.01,
) -> TrainResult:
    """Обучить SpeedPPO на датасете V* (offline PPO / BC-PPO).

    Параметры:
        ppo_eps        — epsilon для clip(ratio, 1-eps, 1+eps).
        ppo_mini_epochs — число мини-эпох PPO на один проход данных.
        c_value        — коэффициент value loss.
        c_entropy      — коэффициент энтропийного бонуса.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    if drone is None:
        drone = QuadModel()

    log.info("PPO — загружаю датасет: %s", csv_path)
    X, y = load_dataset(csv_path)
    N = len(X)
    log.info("  Сэмплов: %d  Признаков: %d", N, X.shape[1])
    train_loader, val_loader, n_train, n_val = _make_loaders(X, y, val_frac, batch_size, seed)
    log.info("  Train: %d  Val: %d", n_train, n_val)

    model: SpeedPPO = get_speed_model("ppo", max_speed=max_speed).to(dev)
    log.info("Модель: %s", model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    es = _ES(patience)

    # Нормализованное advantage (V_opt centred, 0-mean 1-std).
    y_tensor = torch.from_numpy(y.flatten()).float()
    adv_mean = y_tensor.mean().item()
    adv_std = y_tensor.std().item() + 1e-8

    train_losses, val_losses = [], []
    stopped = n_epochs
    t0 = time.monotonic()

    for epoch in range(1, n_epochs + 1):
        # --- Шаг 1: вычислить old log-probs (без градиентов) ---
        model.eval()
        old_log_probs_list = []
        with torch.no_grad():
            for xb, yb in train_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                lp = model.log_prob(xb, yb)
                old_log_probs_list.append(lp.cpu())
        old_log_probs = torch.cat(old_log_probs_list)  # (N_train,)

        # --- Шаг 2: мини-эпохи PPO ---
        model.train()
        tl = 0.0
        for _mini in range(ppo_mini_epochs):
            offset = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                bs = len(xb)

                old_lp = old_log_probs[offset:offset + bs].to(dev)
                offset += bs

                optimizer.zero_grad()

                # Policy
                new_lp = model.log_prob(xb, yb)
                ratio = torch.exp(new_lp - old_lp.detach())

                # Advantage = нормированный V_opt.
                adv = ((yb - adv_mean) / adv_std).detach()

                l_clip = torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1.0 - ppo_eps, 1.0 + ppo_eps) * adv,
                )
                loss_policy = -l_clip.mean()

                # Value
                v_pred = model.forward_value(xb)
                loss_value = mse(v_pred, yb)

                # Entropy bonus
                loss_entropy = -model.entropy(xb).mean()

                loss = loss_policy + c_value * loss_value + c_entropy * loss_entropy
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                tl += loss_value.item() * bs  # отслеживаем value loss как прокси

        train_loss = tl / (n_train * ppo_mini_epochs)
        val_loss = _val_mse(model, val_loader, dev)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 20 == 0 or epoch == 1:
            log.info(
                "Epoch %4d/%d  value_loss=%.5f  val_MSE=%.5f  (%.1fs)",
                epoch, n_epochs, train_loss, val_loss, time.monotonic() - t0,
            )

        if es.step(val_loss, model):
            stopped = epoch
            log.info("Early stop epoch=%d  best_val=%.5f", epoch, es.best)
            break

    es.restore(model)
    save_speed_model_any(model, model_path, drone=drone)
    log.info("Модель сохранена: %s", model_path)

    return TrainResult(
        best_val_loss=es.best,
        train_losses=train_losses,
        val_losses=val_losses,
        stopped_epoch=stopped,
        model_path=model_path,
    )


# ---------------------------------------------------------------------------
# Единая точка входа
# ---------------------------------------------------------------------------

def train_rl(
    model_name: str,
    csv_path: str,
    model_path: str,
    max_speed: float = 10.0,
    n_epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    patience: int = 30,
    seed: int = 42,
    device: str = "cpu",
    drone: Optional[QuadModel] = None,
    **extra,
) -> TrainResult:
    """Универсальная функция обучения — маршрутизирует по model_name.

    model_name: 'mlp' | 'sac' | 'td3' | 'ppo'
    """
    model_name = model_name.lower().strip()

    common = dict(
        csv_path=csv_path,
        model_path=model_path,
        max_speed=max_speed,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        val_frac=val_frac,
        patience=patience,
        seed=seed,
        device=device,
        drone=drone,
    )

    if model_name == "mlp":
        # MLP использует собственный train() из train_model.py.
        from ml.training.train_model import train as train_mlp
        return train_mlp(**common)

    elif model_name == "sac":
        return train_sac(**common, **{k: v for k, v in extra.items()
                                      if k in ("alpha_entropy",)})

    elif model_name == "td3":
        return train_td3(**common, **{k: v for k, v in extra.items()
                                      if k in ("lambda_q", "tau", "actor_delay")})

    elif model_name == "ppo":
        ppo_kw = {k: v for k, v in extra.items()
                  if k in ("ppo_eps", "ppo_mini_epochs", "c_value", "c_entropy")}
        if "lr" not in ppo_kw:
            common["lr"] = min(lr, 3e-4)  # PPO предпочитает меньший lr
        return train_ppo(**common, **ppo_kw)

    else:
        raise ValueError(
            f"Неизвестная модель: {model_name!r}. Доступные: {MODEL_NAMES}"
        )
