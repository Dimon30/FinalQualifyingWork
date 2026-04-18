"""Feature extraction for the speed model."""
from __future__ import annotations

import numpy as np
from typing import Optional

from drone_sim.geometry.curves import CurveGeom, Rz, Ry, se_from_pose
from drone_sim.models.quad_model import QuadModel

# Curvature normalization scale.
_KAPPA_SCALE: float = 1.0

# Lookahead window in parameter space.
_LOOKAHEAD_DS: float = 1.0
_LOOKAHEAD_N: int = 5  # Number of samples in the lookahead window.

# Finite-difference step.
_H: float = 1e-5


def normalize_feature(
    value: float,
    scale: float,
    clip: bool = True,
) -> float:
    """Return ``value / scale`` with optional clipping."""
    if scale < 1e-12:
        return 0.0
    normed = value / scale
    if clip:
        normed = float(np.clip(normed, -1.0, 1.0))
    return float(normed)


def compute_heading_error(
    velocity: np.ndarray,
    s: float,
    curve: CurveGeom,
) -> float:
    """Return the angle between velocity and the curve tangent."""
    v_norm = float(np.linalg.norm(velocity))
    if v_norm < 1e-9:
        # Zero velocity yields a zero heading error.
        return 0.0

    t_vec = curve.t(s)
    t_norm = float(np.linalg.norm(t_vec))
    if t_norm < 1e-9:
        return 0.0

    v_hat = velocity / v_norm
    t_hat = t_vec / t_norm
    dot = float(np.clip(np.dot(v_hat, t_hat), -1.0, 1.0))
    return float(np.arccos(dot))


def compute_kappa(
    s: float,
    curve: CurveGeom,
    h: float = _H,
) -> float:
    """Return the 3D curvature at ``s``."""
    t1 = curve.t(s)
    t2 = (curve.t(s + h) - curve.t(s - h)) / (2.0 * h)  # Approximation of p''(s).

    t1_norm = float(np.linalg.norm(t1))
    if t1_norm < 1e-12:
        return 0.0

    cross = np.cross(t1, t2)
    return float(np.linalg.norm(cross)) / (t1_norm ** 3)


def compute_kappa_max_lookahead(
    s: float,
    curve: CurveGeom,
    lookahead_ds: float = _LOOKAHEAD_DS,
    n_points: int = _LOOKAHEAD_N,
    h: float = _H,
) -> float:
    """Return the maximum curvature on ``[s, s + lookahead_ds]``."""
    points = np.linspace(s, s + lookahead_ds, n_points)
    kappas = [compute_kappa(sp, curve, h) for sp in points]
    return float(np.max(kappas))


def compute_de2_dt(
    velocity: np.ndarray,
    s: float,
    curve: CurveGeom,
) -> float:
    """Return the time derivative of the lateral error component."""
    alpha = curve.yaw_star(s)
    beta_val = curve.beta(s)
    # Projection of the velocity vector to the Frenet frame.
    q = Ry(beta_val).T @ (Rz(alpha).T @ velocity)
    return float(q[2])


def extract_features(
    state: np.ndarray,
    curve: CurveGeom,
    drone: Optional[QuadModel] = None,
    s: float = 0.0,
) -> dict[str, float]:
    """Extract the normalized feature set for the speed model."""
    if drone is None:
        drone = QuadModel()

    state = np.asarray(state, dtype=float)

    # State components used by the feature extractor.
    p_xyz = state[0:3]   # Position.
    velocity = state[3:6]   # Linear velocity [vx, vy, vz].

    # Frenet-frame errors.
    _, e1_raw, e2_raw = se_from_pose(p_xyz, s, curve)

    # Tangential error.
    e1 = normalize_feature(e1_raw, drone.tangential_error_limit)

    # Lateral error.
    e2 = normalize_feature(e2_raw, drone.lateral_error_limit)

    # Time derivative of the lateral error.
    de2_dt_raw = compute_de2_dt(velocity, s, curve)
    de2_dt = normalize_feature(de2_dt_raw, drone.max_velocity_norm)

    # Velocity norm.
    v_norm_raw = float(np.linalg.norm(velocity))
    v_norm = normalize_feature(v_norm_raw, drone.max_speed)

    # Angle between velocity and tangent.
    heading_error_rad = compute_heading_error(velocity, s, curve)
    heading_error = normalize_feature(heading_error_rad, np.pi, clip=False)

    # Curvature at the current point.
    kappa_raw = compute_kappa(s, curve)
    kappa = float(np.clip(kappa_raw / _KAPPA_SCALE, 0.0, 1.0))

    # Maximum curvature on the lookahead window.
    kappa_la_raw = compute_kappa_max_lookahead(s, curve)
    kappa_max_lookahead = float(np.clip(kappa_la_raw / _KAPPA_SCALE, 0.0, 1.0))

    return {
        "e1": e1,
        "e2": e2,
        "de2_dt": de2_dt,
        "v_norm": v_norm,
        "heading_error": heading_error,
        "kappa": kappa,
        "kappa_max_lookahead": kappa_max_lookahead,
    }


def feature_vector(
    state: np.ndarray,
    curve: CurveGeom,
    drone: Optional[QuadModel] = None,
    s: float = 0.0,
) -> np.ndarray:
    """Return the features as a fixed-order NumPy vector."""
    d = extract_features(state, curve, drone=drone, s=s)
    return np.array([
        d["e1"],
        d["e2"],
        d["de2_dt"],
        d["v_norm"],
        d["heading_error"],
        d["kappa"],
        d["kappa_max_lookahead"],
    ], dtype=float)
