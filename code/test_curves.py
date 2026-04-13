"""
test_curves.py — Тесты согласованного управления на базовых кривых.

Запуск:
    python code/test_curves.py           # все тесты (полное время)
    python code/test_curves.py --fast    # быстрый прогон (T/4, только для проверки запуска)

Каждый тест симулирует движение дрона вдоль заданной кривой и оценивает
финальные боковые ошибки ||[e1, e2]||.

Результат: PASS если ||[e1, e2]||_final < PASS_THRESHOLD,  иначе FAIL.
Графики сохраняются в code/out_images/tests/<имя_теста>/

Описание тестов:
    1. spiral_r3         — круговая спираль r=3, из geometry.py (как в диссертации)
    2. circle_r3_z5      — горизонтальная окружность r=3 на z=5, из make_curve
    3. helix_r2          — круговая спираль r=2, из make_curve (промежуточный радиус)
    4. line_diagonal     — прямая x=s,y=s,z=s, аналитическая ближайшая точка
    5. line_xz_plane     — прямая x=s,y=0,z=0.5s, аналитическая ближайшая точка
    6. spiral_r1_5       — спираль r=1.5 (плотные витки, более сложный сценарий)
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from drone_path_sim import make_curve, SimConfig, simulate_path_following
from geometry import spiral_curve, line_xyz_curve, nearest_point_line

# ---------------------------------------------------------------------------
# Аналитическая ближайшая точка для прямой x=s, y=0, z=0.5*s
# (x=s,y=0,z=0.5s): min (zeta-px)^2 + (0.5*zeta-pz)^2
# => 2*(zeta-px) + 0.5*(0.5*zeta-pz) = 0 => 2.5*zeta = 2*px + 0.5*pz => zeta = (2px+0.5pz)/2.5
# ---------------------------------------------------------------------------
def _nearest_line_xz(p_xyz: np.ndarray) -> float:
    """Аналитическая ближайшая точка на прямой p(s)=[s, 0, 0.5*s]."""
    return float((2.0 * p_xyz[0] + 0.5 * p_xyz[2]) / 2.5)


# ---------------------------------------------------------------------------
# Глобальные настройки
# ---------------------------------------------------------------------------
OUT_ROOT = os.path.join(os.path.dirname(__file__), "out_images", "tests")
PASS_THRESHOLD = 1.5   # м — порог финальных боковых ошибок ||[e1,e2]||

FAST_MODE = "--fast" in sys.argv
T_SCALE = 0.25 if FAST_MODE else 1.0


# ===========================================================================
# Описания тестов
# ===========================================================================
#
# Каждый тест — словарь со следующими ключами:
#   name        — короткое имя теста (имя директории с графиками)
#   title       — описание для вывода
#   curve_fn    — callable() -> CurveGeom
#   x0_pos      — начальные координаты дрона [x, y, z]
#   cfg         — словарь параметров SimConfig
#   notes       — (опц.) пояснение по особенностям теста
#
TESTS = [
    # -----------------------------------------------------------------------
    # Тест 1: Круговая спираль r=3 из geometry.py
    # Воспроизводит сценарий из диссертации (стр. 44).
    # Использует NearestPointObserver (gamma=1) — стандартный режим.
    # Ожидание: e1,e2 -> 0, ||v|| -> V*=1, zeta растёт линейно.
    # -----------------------------------------------------------------------
    dict(
        name="spiral_r3",
        title="Круговая спираль r=3 (из geometry.py, как в диссертации)",
        curve_fn=lambda: spiral_curve(r=3.0),
        x0_pos=np.array([2.9, 0.0, 0.0]),   # вблизи кривой (r=3, отступ 0.1 м)
        cfg=dict(
            Vstar=1.0, T=40.0, dt=0.002, kappa=200.0,
            gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0, zeta0=0.0,
        ),
        notes="Baseline: воспроизведение диссертации.",
    ),

    # -----------------------------------------------------------------------
    # Тест 2: Горизонтальная окружность r=3 на высоте z=5
    # Задаётся через make_curve.  z=const -> beta=0 -> 2D задача.
    # Ожидание: дрон делает горизонтальные круги, e1,e2 -> 0.
    # -----------------------------------------------------------------------
    dict(
        name="circle_r3_z5",
        title="Горизонтальная окружность r=3, z=5 (make_curve)",
        curve_fn=lambda: make_curve(
            lambda s: np.array([3.0*np.cos(s), 3.0*np.sin(s), 5.0])
        ),
        x0_pos=np.array([2.8, 0.0, 5.0]),   # вблизи начала окружности
        cfg=dict(
            Vstar=1.0, T=40.0, dt=0.002, kappa=200.0,
            gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0, zeta0=0.0,
        ),
        notes="Горизонтальное движение: z=const, beta=0.",
    ),

    # -----------------------------------------------------------------------
    # Тест 3: Круговая спираль r=2, из make_curve
    # Промежуточный радиус. Проверяет, что make_curve + NearestPointObserver
    # дают тот же результат, что geometry.spiral_curve для другого радиуса.
    #
    # gamma_nearest=3: для r=2 ||t||^2=5. При gamma=1 лаг наблюдателя
    # = V*/(gamma*||t||^2) = 1/5 = 0.2 рад — чуть велик для быстрого схождения.
    # gamma=3 даёт лаг 0.067 рад — достаточно быстро.
    # -----------------------------------------------------------------------
    dict(
        name="helix_r2",
        title="Круговая спираль r=2 (make_curve, промежуточный радиус)",
        curve_fn=lambda: make_curve(
            lambda s: np.array([2.0*np.cos(s), 2.0*np.sin(s), s])
        ),
        x0_pos=np.array([1.9, 0.0, 0.0]),   # вблизи кривой (r=2, отступ 0.1 м)
        cfg=dict(
            Vstar=1.0, T=40.0, dt=0.002, kappa=200.0,
            gamma=(1., 3., 5., 3., 1.), gamma_nearest=3.0, zeta0=0.0,
        ),
        notes="r=2: gamma_nearest=3 (рекомендация: gamma >= 2 / (||t||^2 * dt) при малых r).",
    ),

    # -----------------------------------------------------------------------
    # Тест 4: Диагональная прямая x=s, y=s, z=s из geometry.py
    # Использует аналитическую функцию ближайшей точки (nearest_point_line)
    # вместо NearestPointObserver, чтобы избежать лага наблюдателя.
    #
    # Почему нужна nearest_fn:
    #   NearestPointObserver для прямой имеет лаг ~ v_along / (gamma * ||t||^2).
    #   При ||t||^2=3 и gamma=1 лаг ~ 0.7 рад уже при v=5 м/с,
    #   что создаёт положительную обратную связь и нестабильность.
    # -----------------------------------------------------------------------
    dict(
        name="line_diagonal",
        title="Диагональная прямая x=s,y=s,z=s (nearest_fn=nearest_point_line)",
        curve_fn=lambda: line_xyz_curve(),
        x0_pos=np.array([0.0, 0.0, 0.0]),   # старт в начале прямой
        cfg=dict(
            Vstar=1.0, T=30.0, dt=0.005, kappa=100.0,
            gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0, zeta0=0.0,
            nearest_fn=nearest_point_line,    # аналитическая ближайшая точка
        ),
        notes="Прямая: nearest_fn=nearest_point_line для стабильной симуляции.",
    ),

    # -----------------------------------------------------------------------
    # Тест 5: Прямая в плоскости XZ: p(s)=[s, 0, 0.5*s]
    # Тоже требует аналитической ближайшей точки (NearestPointObserver
    # с gamma=1 нестабилен по той же причине, что и тест 4).
    # -----------------------------------------------------------------------
    dict(
        name="line_xz_plane",
        title="Прямая в плоскости XZ: x=s, y=0, z=0.5*s (nearest_fn аналитическая)",
        curve_fn=lambda: make_curve(
            lambda s: np.array([s, 0.0, 0.5*s])
        ),
        x0_pos=np.array([0.0, 0.1, 0.0]),   # небольшое боковое смещение y=0.1м
        cfg=dict(
            Vstar=1.0, T=30.0, dt=0.002, kappa=200.0,
            gamma=(1., 3., 5., 3., 1.), gamma_nearest=1.0, zeta0=0.0,
            nearest_fn=_nearest_line_xz,     # аналитическая ближайшая точка
        ),
        notes="Наклонная прямая: nearest_fn=(2px+0.5pz)/2.5.",
    ),

    # -----------------------------------------------------------------------
    # Тест 6: Круговая спираль r=1.5 (плотные витки)
    # Более высокая кривизна: curvature=1/r=0.67 (vs 0.33 для r=3).
    # ||t||^2 = r^2+1 = 3.25.
    #
    # gamma_nearest=20: при gamma=1 установившийся лаг наблюдателя
    # = V*/(gamma*||t||^2) = 1/3.25 = 0.31 рад — достаточно большой
    # чтобы вызвать нестабильность при переходных скоростях.
    # gamma=20 даёт лаг 0.015 рад (~0.03 м) — надёжная работа.
    # Правило: gamma_nearest >= 1 / (||t||^2 * dt) = 1/(3.25*0.002) = 154;
    # gamma=20 — разумный компромисс скорость/точность.
    # -----------------------------------------------------------------------
    dict(
        name="spiral_r1_5",
        title="Спираль r=1.5 (плотные витки, gamma_nearest=20)",
        curve_fn=lambda: make_curve(
            lambda s: np.array([1.5*np.cos(s), 1.5*np.sin(s), s])
        ),
        x0_pos=np.array([1.4, 0.0, 0.0]),   # вблизи кривой (r=1.5, отступ 0.1 м)
        cfg=dict(
            Vstar=1.0, T=40.0, dt=0.002, kappa=200.0,
            gamma=(1., 3., 5., 3., 1.), gamma_nearest=20.0, zeta0=0.0,
        ),
        notes="r=1.5: gamma_nearest=20 необходим для устойчивой работы наблюдателя.",
    ),
]


# ===========================================================================
# Запуск одного теста
# ===========================================================================

def run_test(spec: dict) -> dict:
    """Запустить один тест, вернуть словарь с результатами."""
    name = spec["name"]
    cfg_kw = spec["cfg"].copy()

    # Масштаб времени для --fast режима
    cfg_kw["T"] = cfg_kw["T"] * T_SCALE

    # Создать кривую
    curve = spec["curve_fn"]()

    # Начальное состояние
    x0 = np.zeros(16)
    x0[0:3] = spec["x0_pos"]

    cfg = SimConfig(x0=x0, **cfg_kw)

    print(f"  Кривая: {spec['title']}")
    print(f"  T={cfg.T:.0f}с, dt={cfg.dt}, kappa={cfg.kappa}, V*={cfg.Vstar}", end="")
    if cfg.nearest_fn is not None:
        print(f", nearest_fn={cfg.nearest_fn.__name__}", end="")
    print()
    if "notes" in spec:
        print(f"  Замечание: {spec['notes']}")

    result = simulate_path_following(curve, cfg)
    result.plot(
        out_dir=os.path.join(OUT_ROOT, name),
        prefix=name,
    )

    e_final = result.errors[-1]
    v_final = result.velocity[-1]
    lat_err = float(np.linalg.norm(e_final[1:3]))   # ||[e1, e2]||
    passed = lat_err < PASS_THRESHOLD

    print(f"  Финальные ошибки:")
    print(f"    s_arc - V*t = {e_final[0]:+.4f} м")
    print(f"    e1          = {e_final[1]:+.4f} м")
    print(f"    e2          = {e_final[2]:+.4f} м")
    print(f"    delta_phi   = {e_final[3]:+.4f} рад")
    print(f"    ||v||       = {v_final:.4f} м/с  (V* = {cfg.Vstar})")
    print(f"    ||[e1,e2]|| = {lat_err:.4f} м  (порог: {PASS_THRESHOLD} м)")

    status = "PASS" if passed else "FAIL"
    print(f"  --> Статус: {status}")

    return {
        "name": name,
        "title": spec["title"],
        "passed": passed,
        "lat_err": lat_err,
        "e_final": e_final,
        "v_final": v_final,
    }


# ===========================================================================
# Сводная таблица результатов
# ===========================================================================

def print_summary(results: list) -> None:
    passed = sum(r["passed"] for r in results)
    total = len(results)

    print("\n" + "=" * 70)
    print(f"  ИТОГО: {passed}/{total} тестов прошли  (порог ||[e1,e2]|| < {PASS_THRESHOLD} м)")
    print("=" * 70)
    print(f"  {'Тест':<22} {'||[e1,e2]||':>12}  {'||v|| (V*=1)':>13}  {'Статус':>6}")
    print("  " + "-" * 60)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {r['name']:<22} {r['lat_err']:>12.4f}  "
            f"{r['v_final']:>13.4f}  {status:>6}"
        )
    print("=" * 70)
    print(f"  Графики сохранены в {OUT_ROOT}/")
    if FAST_MODE:
        print("  [--fast: T*0.25, переходные процессы могут не завершиться]")


# ===========================================================================
# Точка входа
# ===========================================================================

def main():
    print("=" * 70)
    print("  test_curves.py — тесты согласованного управления по кривой")
    print("=" * 70)
    if FAST_MODE:
        print("  Режим: --fast (T сокращено до 25%)")
    print()

    os.makedirs(OUT_ROOT, exist_ok=True)
    results = []

    for i, spec in enumerate(TESTS, 1):
        print(f"[{i}/{len(TESTS)}] {spec['name']}")
        print("-" * 50)
        try:
            res = run_test(spec)
        except Exception as exc:
            print(f"  ОШИБКА при выполнении теста: {exc}")
            import traceback
            traceback.print_exc()
            res = {
                "name": spec["name"],
                "title": spec["title"],
                "passed": False,
                "lat_err": float("inf"),
                "e_final": np.zeros(4),
                "v_final": 0.0,
            }
        results.append(res)
        print()

    print_summary(results)


if __name__ == "__main__":
    main()
