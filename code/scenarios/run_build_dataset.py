"""
Сборка датасета для обучения SpeedMLP.

Что хранится в датасете:
    Каждая строка — одно состояние дрона на одной кривой:
        - 7 признаков физического состояния (e1, e2, de2_dt, v_norm, ...)
        - V_opt — оптимальная скорость, найденная oracle-поиском
    Геометрия кривой в CSV НЕ хранится — только признаки, вычисленные из неё.
    На каждую кривую берётся --samples точек вдоль параметра s.
    Итого строк ≈ num_curves × samples (часть отсеивается oracle как нестабильные).

Пример запуска (из корня проекта):
    python code/scenarios/build_dataset.py
    python code/scenarios/build_dataset.py --curves 1000 --samples 10
    python code/scenarios/build_dataset.py --curves 200 --coarse-fine --out code/ml/data/my.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_sim.models.quad_model import QuadModel
from ml.config import OracleConfig
from ml.dataset.build_dataset import generate_dataset

# ---------------------------------------------------------------------------
# Выходные директории
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_DEFAULT_CSV = os.path.join(_HERE, "..", "ml", "data", "dataset.csv")
_DEFAULT_OUT_IMG = os.path.join(_HERE, "..", "out_images", "dataset")


def plot_dataset_stats(csv_path: str, out_dir: str) -> None:
    """Нарисовать диагностические графики по готовому CSV."""
    import csv as csvmod

    os.makedirs(out_dir, exist_ok=True)

    rows: list[dict] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    if not rows:
        print("  CSV пуст, графики не строятся.")
        return

    keys = list(rows[0].keys())
    data = {k: np.array([r[k] for r in rows]) for k in keys}
    N = len(rows)

    # --- 1. Распределение V_opt ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(data["V_opt"], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("V_opt")
    axes[0].set_ylabel("Число записей")
    axes[0].set_title(f"Распределение целевой V* (N={N})")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # --- 2. Распределение кривизны ---
    axes[1].hist(data["kappa"], bins=30, color="coral", edgecolor="white")
    axes[1].set_xlabel("kappa (норм.)")
    axes[1].set_title("Распределение кривизны в датасете")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    p = os.path.join(out_dir, "dataset_vopt_kappa.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  График: {p}")

    # --- 3. Матрица рассеяния признаков vs V_opt ---
    feature_cols = ["e1", "e2", "de2_dt", "v_norm",
                    "heading_error", "kappa", "kappa_max_lookahead"]
    present = [c for c in feature_cols if c in data]

    n_f = len(present)
    cols_per_row = 4
    nrows = (n_f + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(nrows, cols_per_row,
                             figsize=(cols_per_row * 4, nrows * 3))
    axes = np.array(axes).flatten()

    for i, col in enumerate(present):
        ax = axes[i]
        ax.scatter(data[col], data["V_opt"],
                   s=4, alpha=0.4, color="steelblue")
        # линия тренда
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

    # Скрыть лишние оси
    for j in range(n_f, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Признаки vs V_opt", fontsize=13)
    fig.tight_layout()
    p = os.path.join(out_dir, "dataset_features_scatter.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  График: {p}")

    # --- 4. Корреляция признаков с V_opt ---
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
    ax.set_xlabel("Корреляция Пирсона с V_opt")
    ax.set_title("Корреляция признаков с целевой переменной")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    fig.tight_layout()
    p = os.path.join(out_dir, "dataset_correlations.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  График: {p}")


def print_csv_summary(csv_path: str) -> None:
    """Вывести статистику датасета в терминал."""
    import csv as csvmod

    rows: list[dict] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    if not rows:
        print("  Датасет пуст!")
        return

    N = len(rows)
    V_opt = np.array([r["V_opt"] for r in rows])
    t_norm = np.array([r["t_norm"] for r in rows])

    print(f"\n{'='*55}")
    print(f"  СТАТИСТИКА ДАТАСЕТА")
    print(f"{'='*55}")
    print(f"  Записей всего     : {N}")
    print(f"  V_opt  min/mean/max: {V_opt.min():.3f} / {V_opt.mean():.3f} / {V_opt.max():.3f}")
    print(f"  t_norm min/mean/max: {t_norm.min():.3f} / {t_norm.mean():.3f} / {t_norm.max():.3f}")

    # Сколько уникальных t_norm (примерно = типы кривых)
    unique_tn = np.unique(np.round(t_norm, 2))
    print(f"  Уникальных ||t||   : {len(unique_tn)} значений")
    print(f"{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Генерация датасета V* для обучения SpeedMLP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--curves", type=int, default=1000,
        help="Число кривых (каждая даёт --samples точек)"
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Точек состояния на одну кривую"
    )
    parser.add_argument(
        "--out", type=str, default=_DEFAULT_CSV,
        help="Путь к выходному CSV"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed для воспроизводимости"
    )
    parser.add_argument(
        "--coarse-fine", action="store_true",
        help="Двухпроходный oracle: шаг 0.5 -> 0.1 (точнее, в ~5x медленнее)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Не строить графики после генерации"
    )
    parser.add_argument(
        "--plots-dir", type=str, default=_DEFAULT_OUT_IMG,
        help="Директория для диагностических графиков"
    )
    args = parser.parse_args()

    # --- Ожидаемый размер датасета ---
    expected = args.curves * args.samples
    print(f"\nГенерация датасета:")
    print(f"  Кривых          : {args.curves}")
    print(f"  Точек на кривую : {args.samples}")
    print(f"  Ожидается строк : ~{expected} (часть отсеивается oracle)")
    print(f"  Oracle режим    : {'coarse-to-fine (0.5->0.1)' if args.coarse_fine else 'линейный (шаг 0.3)'}")
    print(f"  CSV -> {args.out}\n")

    t0 = time.monotonic()
    out_path = generate_dataset(
        num_curves=args.curves,
        out_path=args.out,
        seed=args.seed,
        n_samples_per_curve=args.samples,
        coarse_to_fine=args.coarse_fine,
    )
    elapsed = time.monotonic() - t0

    print(f"\nГотово за {elapsed:.1f} с")
    print_csv_summary(out_path)

    if not args.no_plots:
        print("Строю диагностические графики...")
        plot_dataset_stats(out_path, args.plots_dir)
        print(f"Графики сохранены в {args.plots_dir}")


if __name__ == "__main__":
    main()
