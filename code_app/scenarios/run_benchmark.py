"""Сравнительный бенчмарк всех моделей V* на фиксированном тестовом наборе кривых.

Запуск (из корня проекта)::

    # Полный бенчмарк — все найденные модели, все кривые:
    python code_app/scenarios/run_benchmark.py

    # Только конкретные модели:
    python code_app/scenarios/run_benchmark.py --models mlp,sac

    # Только конкретные кривые:
    python code_app/scenarios/run_benchmark.py --curves spiral_r3,circle_r3z5

    # Указать директорию вывода:
    python code_app/scenarios/run_benchmark.py --out code_app/out_images/benchmark

    # Без построения графиков (только метрики):
    python code_app/scenarios/run_benchmark.py --no-plots

Модели ищутся автоматически в code_app/ml/data/saved_models/:
    mlp -> speed_model.pt
    sac -> sac_model.pt
    td3 -> td3_model.pt
    ppo -> ppo_model.pt

Выходные файлы (code_app/out_images/benchmark/):
    {сценарий}_e2.png           — e2(t): все модели + baseline
    {сценарий}_velocity.png     — v(t)
    summary_e2_rms.png          — grouped bar chart
    summary_v_mean.png
    summary_speedup.png
    summary_table.tex           — LaTeX-таблица (авто-generated)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")

from drone_sim.visualization.plotting import ensure_out, display_path
from ml.evaluation.test_suite import get_test_suite, TestScenario
from ml.evaluation.benchmark import BenchmarkRunner
from ml.evaluation.plots import (
    plot_e2_comparison,
    plot_velocity_comparison,
    plot_summary_bar,
    save_latex_table,
)

# ---------------------------------------------------------------------------
# Пути к моделям
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(__file__)
_MODELS_DIR = os.path.normpath(os.path.join(_HERE, "..", "ml", "data", "saved_models"))
_DEFAULT_OUT = "code_app/out_images/benchmark"

_MODEL_FILE = {
    "mlp": "speed_model.pt",
    "sac": "sac_model.pt",
    "td3": "td3_model.pt",
    "ppo": "ppo_model.pt",
}


def _find_models(wanted: list[str]) -> dict[str, str]:
    """Найти .pt файлы для запрошенных моделей."""
    found: dict[str, str] = {}
    for name in wanted:
        fname = _MODEL_FILE.get(name.lower())
        if fname is None:
            print(f"  [!] Неизвестная модель: {name!r}")
            continue
        full = os.path.join(_MODELS_DIR, fname)
        if os.path.isfile(full):
            found[name] = full
        else:
            print(f"  [!] Не найдена: {full}")
            print(f"       Обучите: python code_app/scenarios/train_rl_model.py --model {name}")
    return found


# ---------------------------------------------------------------------------
# Вывод итоговой таблицы в консоль
# ---------------------------------------------------------------------------

def _print_summary(results, models_run: list[str], scenarios: list[str]) -> None:
    from ml.evaluation.benchmark import ModelResult

    COL = 14
    sep = "=" * (8 + COL * (1 + len(models_run)))
    print(f"\n{sep}")
    print("  БЕНЧМАРК — СВОДНАЯ ТАБЛИЦА (e2_rms, м)")
    print(sep)
    header = f"  {'Сценарий':<22}" + "".join(
        f"  {m.upper():<{COL}}" for m in ["baseline"] + models_run
    )
    print(header)
    print(f"  {'-'*68}")

    for sc in scenarios:
        row = f"  {sc:<22}"
        for m in ["baseline"] + models_run:
            match = [r for r in results
                     if r.scenario_name == sc and r.model_name == m]
            val = f"{match[0].e2_rms:.4f}" if match else "  n/a"
            row += f"  {val:<{COL}}"
        print(row)

    print(f"\n  {'Сценарий':<22}" + "".join(
        f"  {m.upper():<{COL}}" for m in ["baseline"] + models_run
    ))
    print("  " + "  УСКОРЕНИЕ (speedup ×, v_mean)".ljust(68))
    print(f"  {'-'*68}")
    for sc in scenarios:
        row = f"  {sc:<22}"
        for m in ["baseline"] + models_run:
            match = [r for r in results
                     if r.scenario_name == sc and r.model_name == m]
            if match:
                r = match[0]
                val = f"{r.speedup:.2f}×" if m != "baseline" else f"{r.v_mean:.3f} м/с"
            else:
                val = "n/a"
            row += f"  {val:<{COL}}"
        print(row)
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Бенчмарк моделей V* на фиксированном наборе кривых",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", default="mlp,sac,td3,ppo",
        help="Кодовые имена моделей через запятую (mlp,sac,td3,ppo)",
    )
    parser.add_argument(
        "--curves", default="",
        help="Имена сценариев через запятую (по умолчанию — все из тестового набора). "
             "Доступно: spiral_r3, circle_r3z5, helix_r2, line_diag",
    )
    parser.add_argument(
        "--Vstar", type=float, default=1.0,
        help="Базовая V* для режима константной скорости",
    )
    parser.add_argument(
        "--out", default=_DEFAULT_OUT,
        help="Директория для графиков и таблицы",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Не строить графики (только таблица метрик)",
    )
    parser.add_argument(
        "--report_app-images", default="",
        metavar="DIR",
        help="Если указано — дополнительно копировать PNG в эту директорию "
             "(удобно для report_practice/images/)",
    )
    args = parser.parse_args()

    # --- Фильтр сценариев ---
    suite = get_test_suite()
    if args.curves.strip():
        wanted_curves = [c.strip() for c in args.curves.split(",") if c.strip()]
        suite = [s for s in suite if s.name in wanted_curves]
        if not suite:
            print(f"[ОШИБКА] Ни один из указанных сценариев не найден: {wanted_curves}")
            sys.exit(1)

    # --- Загрузка моделей ---
    wanted_models = [m.strip() for m in args.models.split(",") if m.strip()]
    model_paths = _find_models(wanted_models)
    models_run = list(model_paths.keys())

    print(f"\nМодели для бенчмарка: {models_run or '(нет)'}")
    print(f"Сценарии: {[s.name for s in suite]}")
    print(f"Базовая V* = {args.Vstar}")

    ensure_out(args.out)

    # --- Запуск ---
    runner = BenchmarkRunner(
        model_paths=model_paths,
        Vstar_base=args.Vstar,
        verbose=True,
    )
    results = runner.run(suite)
    scenarios = [s.name for s in suite]
    sc_labels  = {s.name: s.label for s in suite}

    # --- Консольная сводка ---
    _print_summary(results, models_run, scenarios)

    # --- Графики ---
    if not args.no_plots:
        print("Строю графики...\n")
        for sc in suite:
            p1 = plot_e2_comparison(
                results, sc.name, args.out, scenario_label=sc.label
            )
            p2 = plot_velocity_comparison(
                results, sc.name, args.out, scenario_label=sc.label,
                Vstar_base=args.Vstar,
            )
            print(f"  {sc.name}: {display_path(p1)}  {display_path(p2)}")

        p = plot_summary_bar(
            results, "e2_rms", args.out,
            ylabel="$e_{2,\\mathrm{RMS}}$, м",
            title="Сравнение: RMS поперечной ошибки $e_2$",
        )
        print(f"  Сводная e2_rms: {display_path(p)}")

        p = plot_summary_bar(
            results, "v_mean", args.out,
            ylabel="$\\bar{v}$, м/с",
            title="Сравнение: средняя скорость",
        )
        print(f"  Сводная v_mean: {display_path(p)}")

        p = plot_summary_bar(
            results, "speedup", args.out,
            ylabel="Ускорение $\\times$",
            title="Сравнение: ускорение относительно базовой $V^*$",
        )
        print(f"  Сводная speedup: {display_path(p)}")

    # --- LaTeX таблица ---
    table_path = os.path.join(args.out, "summary_table.tex")
    save_latex_table(results, table_path)
    print(f"\nLaTeX таблица: {display_path(table_path)}")

    # --- Копирование в директорию отчёта ---
    if args.report_images:
        import shutil
        os.makedirs(args.report_images, exist_ok=True)
        for fname in os.listdir(args.out):
            if fname.endswith(".png"):
                src = os.path.join(args.out, fname)
                dst = os.path.join(args.report_images, fname)
                shutil.copy2(src, dst)
        shutil.copy2(table_path, os.path.join(args.report_images, "summary_table.tex"))
        print(f"Файлы скопированы в: {display_path(args.report_images)}")

    print(f"\nВсё готово. Результаты: {display_path(args.out)}\n")


if __name__ == "__main__":
    main()
