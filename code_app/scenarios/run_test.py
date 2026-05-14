import sys
sys.path.insert(0, 'code')

import numpy as np
from drone_sim import make_curve, SimConfig, QuadModel, simulate_path_following

# Parameterization note:
# for ``p(s) = [r*cos(s), r*sin(s), ...]`` the tangent norm is ``||t|| = r``.
# Large ``r`` requires reparameterization by arc length, e.g. ``cos(s / r)``, ``sin(s / r)``.

# Elliptic spiral: p(s) = [4cos(s), 2sin(s), 0.5s]
# ``||t||^2`` lies in ``[4.25, 16.25]``.
curve = make_curve(lambda s: np.array([4.0 * np.cos(s), 2.0 * np.sin(s), 0.5 * s]))

x0 = np.zeros(16)
x0[0:3] = np.array([4.0, 0.0, 0.0])   # Initial position on the curve.

cfg = SimConfig(
    quad_model=QuadModel(),
    Vstar=1.0,
    T=40.0,
    dt=0.002,
    x0=x0,
    kappa=200.0,
    gamma=(1., 3., 5., 3., 1.),
    gamma_nearest=5.0,   # Bound based on ``2 / (||t||^2_max * dt)``.
    zeta0=0.0,
)

result = simulate_path_following(curve, cfg)
result.print_summary()
result.plot("code/out_images/test", prefix="elliptic")

# Arc-length parameterization example for a large circle:
# curve_big = make_curve(lambda s: np.array([130*np.cos(s/130), 130*np.sin(s/130), 0.0]))
# x0_big = np.zeros(16); x0_big[0:3] = np.array([130.0, 0.0, 0.0])
# cfg_big = SimConfig(Vstar=1.0, T=200.0, dt=0.002, x0=x0_big, gamma_nearest=1.0)
