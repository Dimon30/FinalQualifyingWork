"""
conftest.py корня проекта.

Добавляет code_app/ в sys.path, чтобы `import drone_sim` и `import ml`
работали при запуске pytest из корня проекта:
    pytest code_app/tests/
    pytest code_app/tests/ -v
    pytest code_app/tests/ --fast
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code_app"))
