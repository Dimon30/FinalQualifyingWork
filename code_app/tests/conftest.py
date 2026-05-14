"""
conftest.py — настройка pytest для пакета drone_sim.

Добавляет code_app/ в sys.path, чтобы `import drone_sim` работало
независимо от того, откуда запущен pytest (из code_app/ или из корня проекта).
"""
import sys
import os

# code_app/ — родительская директория этого conftest.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pytest_addoption(parser):
    parser.addoption(
        "--fast", action="store_true", default=False,
        help="Ускоренный прогон: T сокращается до 25%% (для проверки запуска)"
    )
