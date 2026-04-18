"""
conftest.py корня проекта.

Добавляет code/ в sys.path, чтобы `import drone_sim` и `import ml`
работали при запуске pytest из корня проекта:
    pytest code/tests/
    pytest code/tests/ -v
    pytest code/tests/ --fast
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
