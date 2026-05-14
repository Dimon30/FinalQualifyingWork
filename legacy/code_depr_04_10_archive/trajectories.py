from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class TrajPoint4:
    y: np.ndarray
    y1: np.ndarray
    y2: np.ndarray
    y3: np.ndarray
    y4: np.ndarray

def spiral_time_ref(r: float = 0.5) -> Callable[[float], TrajPoint4]:
    """Глава 3: x*=r cos t, y*=r sin t, z*=t, phi*=0."""
    def f(t: float) -> TrajPoint4:
        y  = np.array([r*np.cos(t), r*np.sin(t), t, 0.0], dtype=float)
        y1 = np.array([-r*np.sin(t), r*np.cos(t), 1.0, 0.0], dtype=float)
        y2 = np.array([-r*np.cos(t), -r*np.sin(t), 0.0, 0.0], dtype=float)
        y3 = np.array([r*np.sin(t), -r*np.cos(t), 0.0, 0.0], dtype=float)
        y4 = np.array([r*np.cos(t), r*np.sin(t), 0.0, 0.0], dtype=float)
        return TrajPoint4(y=y,y1=y1,y2=y2,y3=y3,y4=y4)
    return f
