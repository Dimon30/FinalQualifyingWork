"""
ch4_path.py — обратная совместимость.

Все символы перенесены в path_following.py.
Этот модуль сохранён для совместимости с run_ch4_line.py, run_ch4_spiral.py
и другими скриптами, которые импортируют из controllers.ch4_path.
"""
from controllers.path_following import (  # noqa: F401
    W_mat,
    W_inv,
    b_mat,
    _safe_inv4,
    Ch4PathController,
)
