# src/drone/trajectories.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Protocol, Tuple


class Trajectory(Protocol):
    """
    Trajectory parametrized by s.
    Returns r(s) and dr/ds (needed if we want constant spatial speed V*).
    """
    def eval(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        ...


@dataclass
class LineXYZ:
    """
    r(s) = [s, s, s]
    """
    def eval(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        r = np.array([s, s, s], dtype=float)
        dr = np.array([1.0, 1.0, 1.0], dtype=float)
        return r, dr


@dataclass
class Helix:
    """
    r(s) = [R*cos(w*s), R*sin(w*s), vz*s]
    """
    R: float = 3.0
    w: float = 1.0
    vz: float = 0.3

    def eval(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        r = np.array([self.R * np.cos(self.w * s),
                      self.R * np.sin(self.w * s),
                      self.vz * s], dtype=float)
        dr = np.array([-self.R * self.w * np.sin(self.w * s),
                        self.R * self.w * np.cos(self.w * s),
                        self.vz], dtype=float)
        return r, dr


@dataclass
class CircleXY:
    """
    r(s) = [R*cos(w*s), R*sin(w*s), z0]
    """
    R: float = 3.0
    w: float = 1.0
    z0: float = 5.0

    def eval(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        r = np.array([self.R * np.cos(self.w * s),
                      self.R * np.sin(self.w * s),
                      self.z0], dtype=float)
        dr = np.array([-self.R * self.w * np.sin(self.w * s),
                        self.R * self.w * np.cos(self.w * s),
                        0.0], dtype=float)
        return r, dr


def make_trajectory(name: str) -> Trajectory:
    """
    Factory: choose trajectory by name. Extend here only.
    """
    name = name.lower().strip()
    if name in ("line", "linexyz", "xyz"):
        return LineXYZ()
    if name in ("helix", "spiral", "spira"):
        return Helix(R=3.0, w=1.0, vz=0.3)
    if name in ("circle", "circlexy"):
        return CircleXY(R=3.0, w=1.0, z0=5.0)

    raise ValueError(f"Unknown trajectory '{name}'. Available: line, helix, circle")