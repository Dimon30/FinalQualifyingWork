# Исследование методов интеллектуального управления траекторным движением в 3D для класса нелинейных систем

Выпускная квалификационная работа по теме **«Исследование методов интеллектуального управления траекторным движением в трёхмерном пространстве для класса нелинейных систем»**.
Объект управления — квадрокоптер (БПЛА).

Математическая основа: диссертация Ким С.А. «Алгоритмы траекторного согласованного управления по выходу для класса нелинейных систем» (2024).

---

## Описание

Проект реализует алгоритм **согласованного управления по выходу** (Глава 4 диссертации):
движение квадрокоптера вдоль произвольной пространственной кривой с заданной скоростью.

Весь код оформлен в виде Python-пакета **`drone_sim`**, который предоставляет:
- задание физических параметров дрона (`QuadModel`)
- задание произвольной кривой (`make_curve`)
- настройку и запуск симуляции (`SimConfig`, `simulate_path_following`)
- анализ результатов (`SimResult`)

---

## Быстрый старт

```bash
pip install numpy matplotlib pytest

# Сценарии из диссертации (Гл. 4)
python code/scenarios/run_ch4_line.py     # согласованное управление (прямая)
python code/scenarios/run_ch4_spiral.py   # согласованное управление (спираль r=3)

# Тесты (6 кривых, PASS/FAIL)
pytest code/tests/
pytest code/tests/ --fast    # ускоренный прогон (T×0.25)
```

Графики сохраняются в `code/out_images/`.

---

## Использование пакета drone_sim

Пакет `drone_sim` находится в директории `code/`. Чтобы импорты работали,
**запускайте скрипты из директории `code/`** или добавьте её в `PYTHONPATH`:

```bash
# Вариант 1: запускать из code/
cd code
python my_script.py

# Вариант 2: указать PYTHONPATH при запуске из корня
PYTHONPATH=code python my_script.py   # Linux/macOS
$env:PYTHONPATH="code"; python my_script.py  # Windows PowerShell
```

После этого во всех скриптах:

```python
from drone_sim import make_curve, SimConfig, simulate_path_following
```

### Шаг 1. Задать параметры дрона

```python
from drone_sim import QuadModel

# Нормализованная модель диссертации (значения по умолчанию)
model = QuadModel()

# Явные физические параметры
model = QuadModel(
    g=9.81,       # ускорение свободного падения [м/с²]
    mass=1.5,     # масса [кг]
    J_phi=0.02,   # момент инерции: рысканье [кг·м²]
    J_theta=0.02, # момент инерции: тангаж   [кг·м²]
    J_psi=0.04,   # момент инерции: крен     [кг·м²]
)
```

> При изменении `mass` или `J` регуляторные коэффициенты (`kappa`, `gamma`) требуют
> перенастройки — они нормированы под нормализованную модель (mass=1, J=1).

### Шаг 2. Задать кривую

```python
import numpy as np
from drone_sim import make_curve

# Круговая спираль: x=r·cos(s), y=r·sin(s), z=s
curve = make_curve(lambda s: np.array([3.0*np.cos(s), 3.0*np.sin(s), s]))

# Эллиптическая спираль
curve = make_curve(lambda s: np.array([4.0*np.cos(s), 2.0*np.sin(s), 0.5*s]))

# Горизонтальная окружность на высоте 5 м
curve = make_curve(lambda s: np.array([3.0*np.cos(s), 3.0*np.sin(s), 5.0]))

# Параболическая кривая
curve = make_curve(lambda s: np.array([s, 0.3*s**2, 0.2*s]))
```

`make_curve` автоматически вычисляет геометрические характеристики
через численное дифференцирование: касательный вектор, углы рысканья и тангажа, кривизну.

Готовые кривые:

```python
from drone_sim.geometry import spiral_curve, line_xyz_curve

curve = spiral_curve(r=3.0)   # спираль r=3 (как в диссертации)
curve = line_xyz_curve()      # прямая x=s, y=s, z=s
```

### Шаг 3. Настроить параметры симуляции

```python
from drone_sim import SimConfig, QuadModel

cfg = SimConfig(
    quad_model=QuadModel(),  # None → нормализованная модель

    Vstar=1.0,           # желаемая параметрическая скорость [м/с]
    T=30.0,              # время симуляции [с]
    dt=0.002,            # шаг интегрирования RK4 [с]
    x0=None,             # начальное состояние 16D (None → старт на кривой в zeta0)

    kappa=200.0,
    a=(5., 10., 10., 5., 1.),
    gamma=(1., 3., 5., 3., 1.),
    L=5.0,
    ell=0.9,

    gamma_nearest=1.0,   # коэффициент наблюдателя ближайшей точки
    zeta0=0.0,
)
```

**Задание начального положения:**

```python
import numpy as np

# Вариант А: старт на кривой в точке zeta0 (по умолчанию)
cfg = SimConfig(x0=None, zeta0=0.0)

# Вариант Б: произвольное начальное положение
x0 = np.zeros(16)
x0[0:3] = np.array([2.9, 0.0, 0.0])   # [x, y, z]
cfg = SimConfig(x0=x0)
```

Структура вектора состояния 16D:
```
x[0:3]   = [x, y, z]                   — положение
x[3:6]   = [vx, vy, vz]                — скорость
x[6:9]   = [phi, theta, psi]           — рысканье, тангаж, крен
x[9:12]  = [phidot, thetadot, psidot]  — угловые скорости
x[12:14] = [u1_bar, rho1]              — интегратор тяги
x[14:16] = [u2, rho2]                  — интегратор рысканья
```

**Подбор kappa и dt:**

| kappa | Рекомендуемый dt | Применение |
|-------|-----------------|------------|
| 100   | ≤ 0.010 с       | Простые кривые, тестирование |
| 200   | ≤ 0.005 с       | Стандарт диссертации (Гл. 4) |
| 300   | ≤ 0.002 с       | Высокая точность/кривизна |

### Шаг 4. Запустить симуляцию

```python
from drone_sim import simulate_path_following

result = simulate_path_following(curve, cfg)
```

### Шаг 5. Анализ результатов

```python
result.print_summary()                          # финальные ошибки в консоль
result.plot("out_images/my_curve", prefix="r")  # сохранить 6 графиков
```

| Файл | Содержание |
|------|-----------|
| `r_traj_3d.png` | 3D траектория дрона и заданной кривой |
| `r_traj_xy.png` | Проекция X-Y |
| `r_errors.png` | Ошибки: `s_arc − V*·t`, `e1`, `e2` |
| `r_yaw_error.png` | Ошибка рысканья `δφ` |
| `r_velocity.png` | Скорость `‖v‖` и желаемая `V*` |
| `r_angles.png` | Угловые координаты φ, θ, ψ |

Числовые данные:

```python
result.t          # массив времени [n]
result.x          # траектория состояния [n×16]
result.p_ref      # ближайшие точки на кривой [n×3]
result.errors     # [n×4]: s_arc-V*t, e1, e2, delta_phi
result.velocity   # норма скорости [n]
result.zeta       # параметр ближайшей точки [n]
```

### Полный пример

```python
import numpy as np
from drone_sim import make_curve, SimConfig, QuadModel, simulate_path_following

# Наклонная эллиптическая спираль
curve = make_curve(lambda s: np.array([4.0*np.cos(s), 2.0*np.sin(s), 0.5*s]))

x0 = np.zeros(16)
x0[0:3] = np.array([3.8, 0.0, 0.0])

cfg = SimConfig(
    quad_model=QuadModel(),
    Vstar=1.0, T=40.0, dt=0.002, x0=x0, kappa=200.0,
)

result = simulate_path_following(curve, cfg)
result.print_summary()
result.plot("out_images/elliptic_helix")
```

### Рекомендации и ограничения

**Равномерная параметризация.**
Для точности `s_arc = ζ·‖t(ζ)‖` рекомендуются кривые с постоянной нормой касательного
вектора (`‖dp/ds‖ = const`). Стандартные кривые (спираль, прямая) удовлетворяют этому.

**Выбор gamma_nearest.**
```python
# Условие устойчивости дискретного наблюдателя:
#   0 < gamma * dt * ||t||^2_max < 2
#   => gamma < 2 / (||t||^2_max * dt)
#
# Для равномерных спиралей ||t||^2 = r^2 + 1 = const:
#   r=3:   ||t||^2=10,   gamma < 100,  используем gamma=1
#   r=2:   ||t||^2=5,    gamma < 200,  используем gamma=3
#   r=1.5: ||t||^2=3.25, gamma < 308,  используем gamma=20
#
# Для неравномерных кривых (эллипс, парабола) ||t||^2 меняется —
# ограничение считается по МАКСИМУМУ ||t||^2:
#   эллипс [4cos,2sin,0.5s]: ||t||^2 in [4.25, 16.25] => gamma < 61.5, берём gamma=5
```

**Прямые кривые.**
Для прямых `NearestPointObserver` нестабилен — используйте аналитическую `nearest_fn`:
```python
from drone_sim.geometry import nearest_point_line

cfg = SimConfig(..., nearest_fn=nearest_point_line)
```

**Начальное положение.** Дрон должен стартовать вблизи кривой (≲ 1–2 м).

---

## Тесты

```bash
pytest code/tests/           # все 6 тестов
pytest code/tests/ -v        # с именами
pytest code/tests/ -k helix  # по маске имени
pytest code/tests/ --fast    # быстрый прогон (T×0.25)
```

Тесты запускают 6 сценариев (спираль r=3, окружность, спираль r=2, прямые, спираль r=1.5),
сохраняют графики в `code/out_images/tests/` и проверяют `‖[e1,e2]‖_final < 1.5 м`.

---

## Структура проекта

```
code/
  drone_sim/            — Python-пакет
    models/             — QuadModel, динамика (quad_dynamics_16, sat_tanh)
    geometry/           — CurveGeom, кривые, ошибки Френе
    control/            — HighGainParams, Ch4PathController, W_mat, b_mat
    simulation/         — rk4_step, simulate, PathFollowingController, SimConfig, SimResult
    visualization/      — функции построения графиков
    nn/                 — placeholder: нейросетевые алгоритмы V*
  scenarios/
    run_ch4_line.py     — сценарий: прямая (диссертация стр. 41)
    run_ch4_spiral.py   — сценарий: спираль r=3 (диссертация стр. 43–44)
  tests/
    test_curves.py      — pytest: 6 кривых

legacy/                 — архив симуляций Гл. 2–3 (не трогать)
report/                 — LaTeX-исходники отчёта
Диссертация на сайт.pdf — математическая основа
```

---

## Математическая модель

### Динамика квадрокоптера (уравнения 52–55 диссертации)

**Соглашение об углах** (нестандартное): φ = рысканье (yaw), θ = тангаж (pitch), ψ = крен (roll).

```
ṗ = v
v̇ = (b(φ,θ,ψ)·(u1 + g) − [0, 0, g]) / mass
φ̈ = u2 / J_phi,   θ̈ = u3 / J_theta,   ψ̈ = u4 / J_psi
```

Цепочки двойных интеграторов (расширение состояния):
```
ρ̇1 = v1,   u̇1_bar = ρ1,   u1 = L·tanh(u1_bar / L)   — тяга
ρ̇2 = v2,   u̇2 = ρ2                                   — рысканье
```

### Алгоритм согласованного управления (Гл. 4)

1. **Наблюдатель ближайшей точки** (Лемма 3):
   `ζ̇ = −γ · sign(∂H/∂ζ) · H(ζ, x)`,   `H = (p(ζ) − x) · t(ζ)`

2. **Ошибки в системе Френе** (ур. 60):
   `λ̃₁ = [s_arc − V*t,  e₁,  e₂,  δφ]`

3. **Наблюдатель производных** (ур. 73–77):
   пятиступенчатый наблюдатель с коэффициентом κ оценивает `λ̃₁` и её производные до 4-го порядка.

4. **Закон управления** (ур. 71–72):
   `Ū = sat_L[b⁻¹ W⁻¹ (−σ − Σᵢ γᵢ λ̂ᵢ)]`,   `U = γ₅·η + Ū`,   `η̇ = Ū`

---

## Зависимости

- Python 3.9+
- numpy, matplotlib, pytest

```bash
pip install numpy matplotlib pytest
# или
pip install -r requirements.txt
```

---

## Планируемые расширения

- Замена константного `V*` на нейросетевой алгоритм (RL: PPO/SAC) для адаптивного
  выбора оптимальной скорости движения вдоль траектории. Заготовка: `drone_sim/nn/`.
