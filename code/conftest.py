"""
conftest.py корневой для code/.

Добавляет code/ в sys.path, чтобы `import drone_sim` работало при запуске
pytest как из code/, так и из корня проекта.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
