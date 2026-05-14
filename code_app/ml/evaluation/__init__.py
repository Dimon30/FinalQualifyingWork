"""Модуль оценки и сравнительного анализа моделей предсказания V*.

Публичный API::

    from ml.evaluation.test_suite import get_test_suite
    from ml.evaluation.benchmark import BenchmarkRunner, ModelResult
    from ml.evaluation.plots import plot_e2_comparison, plot_velocity_comparison
    from ml.evaluation.plots import plot_summary_bar, save_latex_table
"""
from ml.evaluation.test_suite import TestScenario, get_test_suite
from ml.evaluation.benchmark import ModelResult, BenchmarkRunner

__all__ = [
    "TestScenario",
    "get_test_suite",
    "ModelResult",
    "BenchmarkRunner",
]
