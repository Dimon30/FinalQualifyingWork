"""
test_curves.py — pytest тесты согласованного управления на базовых кривых.

Запуск:
    pytest code_app/tests/                        # все тесты (из корня проекта)
    pytest code_app/tests/ -v                     # с именами тестов
    pytest code_app/tests/ -k spiral_r3          # конкретный тест
    pytest code_app/tests/ --fast                 # ускоренный прогон (T×0.25)
    pytest code_app/tests/ -s                     # с выводом print

Каждый тест симулирует движение дрона вдоль заданной кривой и проверяет,
что финальные боковые ошибки ||[e1, e2]|| < PASS_THRESHOLD.

Описание сценариев:
    spiral_r3       — круговая спираль r=3 (как в диссертации)
    circle_r3_z5    — горизонтальная окружность r=3 на z=5
    helix_r2        — круговая спираль r=2 (промежуточный радиус)
    line_diagonal   — прямая x=s,y=s,z=s с аналитической nearest_fn
    line_xz_plane   — прямая x=s,y=0,z=0.5s с аналитической nearest_fn
    spiral_r1_5     — спираль r=1.5 (плотные витки, gamma_nearest=20)
"""
import os
import sys
import pytest
import numpy as np

from drone_sim import make_curve, SimConfig, simulate_path_following
from drone_sim.geometry.curves import spiral_curve, line_xyz_curve, nearest_point_line

# ---------------------------------------------------------------------------
# Порог прохождения теста
# ---------------------------------------------------------------------------
PASS_THRESHOLD = 1.5   # м — ||[e1, e2]||_final

# ---------------------------------------------------------------------------
# Аналитическая ближайшая точка для прямой p(s)=[s, 0, 0.5*s]
# ---------------------------------------------------------------------------
def _nearest_line_xz(p_xyz: np.ndarray) -> float:
    return float((2.0 * p_xyz[0] + 0.5 * p_xyz[2]) / 2.5)


# ---------------------------------------------------------------------------
# Описания тестовых сценариев
# ---------------------------------------------------------------------------
TESTS = [
    dict(
        name="spiral_r3",
        title="Круговая спираль r=3 (baseline из диссертации)",
        curve_fn=lambda: spiral_curve(r=3.0),
        x0_pos=np.array([2.9, 0.0, 0.0]),
        cfg=dict(Vstar=1.0, T=40.0, dt=0.002, kappa=200.0,
                 gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0, zeta0=0.0),
    ),
    dict(
        name="circle_r3_z5",
        title="Горизонтальная окружность r=3, z=5",
        curve_fn=lambda: make_curve(
            lambda s: np.array([3.0*np.cos(s), 3.0*np.sin(s), 5.0])
        ),
        x0_pos=np.array([3.0, 0.0, 5.0]),
        cfg=dict(Vstar=1.0, T=40.0, dt=0.002, kappa=200.0,
                 gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0, zeta0=0.0),
    ),
    dict(
        name="helix_r2",
        title="Круговая спираль r=2 (gamma_nearest=3)",
        curve_fn=lambda: make_curve(
            lambda s: np.array([2.0*np.cos(s), 2.0*np.sin(s), s])
        ),
        x0_pos=np.array([1.9, 0.0, 0.0]),
        cfg=dict(Vstar=1.0, T=40.0, dt=0.002, kappa=200.0,
                 gamma=(1., 3., 5., 3., 1.), gamma_nearest=3.0, zeta0=0.0),
    ),
    dict(
        name="line_diagonal",
        title="Прямая x=s,y=s,z=s (nearest_fn=nearest_point_line)",
        curve_fn=lambda: line_xyz_curve(),
        x0_pos=np.array([0.0, 0.0, 0.0]),
        cfg=dict(Vstar=1.0, T=30.0, dt=0.005, kappa=100.0,
                 gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0, zeta0=0.0,
                 nearest_fn=nearest_point_line),
    ),
    # TODO перепроверить алгоритм, есть подозрение что написано неправильно
    dict(
        name="line_xz_plane",
        title="Прямая x=s,y=0,z=0.5s (аналитическая nearest_fn)",
        curve_fn=lambda: make_curve(lambda s: np.array([s, 0.0, 0.5*s])),
        x0_pos=np.array([0.0, 0.1, 0.0]),
        cfg=dict(Vstar=1.0, T=30.0, dt=0.002, kappa=200.0,
                 gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0, zeta0=0.0,
                 nearest_fn=_nearest_line_xz),
    ),
    dict(
        name="spiral_r1_5",
        title="Спираль r=1.5 (плотные витки, gamma_nearest=20)",
        curve_fn=lambda: make_curve(
            lambda s: np.array([1.5*np.cos(s), 1.5*np.sin(s), s])
        ),
        x0_pos=np.array([1.4, 0.0, 0.0]),
        cfg=dict(Vstar=1.0, T=40.0, dt=0.002, kappa=200.0,
                 gamma=(1., 3., 5., 3., 1.), gamma_nearest=20.0, zeta0=0.0),
    ),
]


# ---------------------------------------------------------------------------
# Фикстура: опция --fast
# ---------------------------------------------------------------------------
@pytest.fixture
def fast_mode(request):
    return request.config.getoption("--fast")


# ---------------------------------------------------------------------------
# Директория для графиков
# ---------------------------------------------------------------------------
_OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "out_images", "tests")


# ---------------------------------------------------------------------------
# Параметризованный тест
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec", TESTS, ids=[t["name"] for t in TESTS])
def test_path_following(spec, fast_mode):
    """Проверить, что ||[e1, e2]||_final < PASS_THRESHOLD для каждой кривой."""
    cfg_kw = spec["cfg"].copy()

    if fast_mode:
        cfg_kw["T"] = cfg_kw["T"] * 0.25

    curve = spec["curve_fn"]()

    x0 = np.zeros(16)
    x0[0:3] = spec["x0_pos"]

    cfg = SimConfig(x0=x0, **cfg_kw)

    result = simulate_path_following(curve, cfg)

    # Сохранить графики
    out_dir = os.path.join(_OUT_ROOT, spec["name"])
    result.plot(out_dir=out_dir, prefix=spec["name"])

    e_final = result.errors[-1]
    lat_err = float(np.linalg.norm(e_final[1:3]))

    print(f"\n  {spec['title']}")
    print(f"    e1={e_final[1]:+.4f} м,  e2={e_final[2]:+.4f} м")
    print(f"    ||[e1,e2]|| = {lat_err:.4f} м  (порог: {PASS_THRESHOLD} м)")
    print(f"    ||v|| = {result.velocity[-1]:.4f} м/с")

    assert lat_err < PASS_THRESHOLD, (
        f"[{spec['name']}] ||[e1,e2]|| = {lat_err:.4f} м >= {PASS_THRESHOLD} м"
    )
