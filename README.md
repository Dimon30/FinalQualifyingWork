# Исследование методов интеллектуального управления траекторным движением в 3D

**Выпускная квалификационная работа**
«Исследование методов интеллектуального управления траекторным движением в трёхмерном пространстве для класса нелинейных систем»

Математическая основа: диссертация Ким С.А. «Алгоритмы траекторного согласованного управления по выходу для класса нелинейных систем» (2024).

---

## Оглавление

1. [Описание](#описание)
2. [Зависимости](#зависимости)
3. [Быстрый старт](#быстрый-старт)
4. [Структура проекта](#структура-проекта)
5. [Математическая основа](#математическая-основа)
6. [Использование пакета drone\_sim](#использование-пакета-drone_sim)
7. [ML-пайплайн](#ml-пайплайн)
8. [Рекомендации и ограничения](#рекомендации-и-ограничения)
9. [Планируемые расширения](#планируемые-расширения)

---

## Описание

В работе решаются следующие задачи:

1. **Постановка задачи траекторного управления** — формализация задачи движения нелинейной динамической системы вдоль заданной кривой в трёхмерном пространстве с заданной скоростью.

2. **Базовый алгоритм траекторного слежения** — реализация алгоритма согласованного управления по выходу (Гл. 4 диссертации) на основе метода высокого усиления. Движение квадрокоптера вдоль произвольной пространственной кривой с наблюдателем ближайшей точки.

3. **Доработка модели движения** — реализация 16-мерного вектора состояния с двойными интеграторными расширениями по тяге и рысканью; учёт пространственной кинематики через матрицы поворота Фрейне; насыщение тяги через `sat_tanh`.

4. **Методы обучения с подкреплением для оптимизации скорости** — исследование подходов к адаптивному выбору параметрической скорости V*. На текущем этапе реализован supervised-подход (oracle + SpeedMLP); RL-подход (PPO/SAC) запланирован как следующий шаг.

5. **Сравнительный анализ** — сопоставление базовой симуляции с константной V* и нейросетевого оптимизатора по метрикам ошибки слежения (e1, e2, RMS) и средней скорости движения.

---

## Зависимости

| Библиотека | Назначение |
|---|---|
| `numpy` | Математические операции: матричные вычисления, численное интегрирование, обработка траекторий |
| `matplotlib` | Визуализация: 3D-траектории, графики ошибок, скоростей, углов |
| `pytest` | Автоматизированное тестирование: запуск 6 сценариев слежения с проверкой порога ошибок |
| `torch` (PyTorch) | ML-пайплайн: обучение SpeedMLP, сохранение/загрузка чекпоинтов, инференс |

Установка:

```bash
# Основные зависимости:
pip install -r requirements.txt

# ML-пайплайн (дополнительно):
pip install torch

# Пакет drone_sim в режиме разработки (рекомендуется):
pip install -e code/
```

---

## Быстрый старт

Все команды выполняются из **корня проекта**.

### 1. Подготовка окружения

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install torch
pip install -e code/
```

### 2. Базовые сценарии из Главы 4 диссертации

```bash
# Движение вдоль прямой (x=s, y=s, z=s):
python code/scenarios/run_ch4_line.py

# Движение вдоль спирали r=3:
python code/scenarios/run_ch4_spiral.py
```

Графики сохраняются в `code/out_images/ch4_line/` и `code/out_images/ch4_spiral/`.

### 3. Запуск тестов

```bash
# Базовые кривые (круговая спираль + прямые):
pytest code/tests/ -k "spiral_r3 or line_diagonal or helix_r2" -v

# Все 6 тестов в ускоренном режиме (T×0.25):
pytest code/tests/ --fast -v

# Полный прогон (занимает несколько минут):
pytest code/tests/ -v
```

Результаты сохраняются в `code/out_images/tests/{имя_теста}/`.

> **Примечание.** В быстром режиме (`--fast`) время симуляции сокращается до 25%, что может не дать достаточно времени наблюдателю для сходимости у кривых с высокой кривизной. Для надёжного прогона используйте полный режим или выбирайте базовые кривые.

### 4. Тестовый запуск дрона с произвольной кривой

```bash
# Базовая симуляция (эллиптическая спираль, константная V*):
python code/scenarios/run_test_drone.py

# С NN-оптимизатором (требует обученной модели, см. шаги 6–7):
python code/scenarios/run_test_drone.py --model auto

# Указать собственную директорию для результатов:
python code/scenarios/run_test_drone.py --out code/out_images/my_experiment
```

Графики по умолчанию сохраняются в `code/out_images/test_drone/`.

### 5. Сборка небольшого датасета

```bash
python code/scenarios/run_build_dataset.py --curves 10 --samples 20
```

Датасет сохраняется в `code/ml/data/dataset.csv`.
Диагностические графики — в `code/out_images/dataset/`.

### 6. Обучение базовой MLP-модели

```bash
python code/scenarios/train_speed_model.py --epochs 100 --patience 15
```

Модель сохраняется в `code/ml/data/model/speed_model.pt`.
Графики качества — в `code/out_images/training/`.

### 7. Инференс: симуляция с NN-оптимизатором

```bash
# Стандартный путь (code/ml/data/model/speed_model.pt):
python code/scenarios/run_test_drone.py --model default

# Авто-поиск последней модели:
python code/scenarios/run_test_drone.py --model auto
```

### 8. Сравнение: константная V* vs NN

```bash
# На спирали r=3:
python code/scenarios/run_nn_speed.py --curve spiral

# На прямой:
python code/scenarios/run_nn_speed.py --curve line

# Сохранить в отдельную директорию:
python code/scenarios/run_nn_speed.py --curve spiral --out code/out_images/compare_spiral
```

Сравнительные графики сохраняются в `code/out_images/nn_speed/` (по умолчанию).

---

## Структура проекта

```
code/
  drone_sim/                  — основной Python-пакет симуляции
    __init__.py               — публичный API: make_curve, SimConfig, simulate_path_following
    models/
      quad_model.py           — QuadModel: физика дрона + лётные ограничения для ML
      dynamics.py             — quad_dynamics_16, quad_dynamics_12, sat_tanh, матрица G
    geometry/
      curves.py               — CurveGeom, line_xyz_curve, spiral_curve, nearest_point_line
    control/
      common.py               — HighGainParams, DerivativeObserver4
      path_following.py       — Ch4PathController, W_mat, W_inv, b_mat, _safe_inv4
    simulation/
      integrators.py          — rk4_step
      runner.py               — simulate (цикл RK4 для низкоуровневых сценариев)
      path_sim.py             — NearestPointObserver, PathFollowingController,
                                SimConfig, SimResult, simulate_path_following
    visualization/
      plotting.py             — plot_3d_traj, plot_errors, plot_velocity, plot_angles, plot_xy
    nn/
      __init__.py             — placeholder: будущие RL-алгоритмы V*

  ml/                         — ML-пайплайн оптимизации V* (supervised)
    config.py                 — MLConfig, OracleConfig, DEFAULT_MODEL_PATH
    curves/
      generator.py            — CurveSpec, make_line/circle/spiral_curve
    dataset/
      build_dataset.py        — generate_dataset: oracle + запись CSV
      features.py             — extract_features, feature_vector (7 признаков)
      simulator_wrapper.py    — rollout_with_speed, is_stable, find_optimal_speed
    models/
      speed_model.py          — SpeedMLP (128→128→64→1), save/load_speed_model
    training/
      train_model.py          — train() с early stopping, TrainResult
    inference/
      predict.py              — SpeedPredictor: load/predict/default
    data/                     — dataset.csv, model/speed_model.pt  ← не коммитить

  scenarios/
    run_ch4_line.py           — Гл. 4 диссертации: прямая (стр. 41)
    run_ch4_spiral.py         — Гл. 4 диссертации: спираль r=3 (стр. 43–44)
    run_test_drone.py         — тестовый запуск дрона с произвольной кривой (+ NN)
    run_build_dataset.py      — сборка датасета (oracle V*)
    train_speed_model.py      — обучение SpeedMLP + графики качества
    run_nn_speed.py           — сравнение: константная V* vs NN-оптимизатор

  tests/
    conftest.py               — sys.path + --fast hook
    test_curves.py            — pytest: 6 кривых, PASS/FAIL по ||[e1,e2]|| < 1.5 м
  conftest.py                 — sys.path для запуска pytest из корня
  pytest.ini                  — testpaths = tests, pythonpath = .

code/out_images/              — результаты симуляций (не коммитить)
  ch4_line/                   — сценарий прямой
  ch4_spiral/                 — сценарий спирали
  test_drone/                 — тестовый запуск run_test_drone.py
  tests/{имя_теста}/          — pytest результаты
  dataset/                    — диагностика датасета
  training/                   — графики обучения
  nn_speed/                   — сравнение baseline vs NN

legacy/                       — архив симуляций Гл. 2–3 (не трогать)
report/                       — LaTeX-исходники отчёта
Диссертация на сайт.pdf       — математическая основа
requirements.txt
```

---

## Математическая основа

### Симуляция дрона

#### Вектор состояния (16-мерный, Гл. 3–4)

```
x = [x, y, z,  vx, vy, vz,  φ, θ, ψ,  φ̇, θ̇, ψ̇,  u1_bar, ρ1, u2, ρ2]
```

| Индексы | Переменные | Смысл |
|---|---|---|
| `x[0:3]` | x, y, z | Положение дрона в пространстве |
| `x[3:6]` | vx, vy, vz | Линейная скорость |
| `x[6:9]` | φ, θ, ψ | Рысканье (yaw), тангаж (pitch), крен (roll) |
| `x[9:12]` | φ̇, θ̇, ψ̇ | Угловые скорости |
| `x[12:14]` | u1_bar, ρ1 | Состояние двойного интегратора тяги |
| `x[14:16]` | u2, ρ2 | Состояние двойного интегратора по рысканью |

> **Соглашение об углах** (нестандартное): φ = рысканье (yaw), θ = тангаж (pitch), ψ = крен (roll).

#### Уравнения движения

```
ṗ = v
v̇ = (b(φ,θ,ψ)·u1 − [0, 0, g]) / mass
φ̈ = u2/J_φ,   θ̈ = u3/J_θ,   ψ̈ = u4/J_ψ
```

Расширение состояния (двойные интеграторы):
```
ρ̇1 = v1,   u̇1_bar = ρ1,   u1 = L·tanh(u1_bar / L)   — насыщение тяги
ρ̇2 = v2,   u̇2 = ρ2                                   — интегратор по рысканью
```

Управляющий вход: `U = [v1, v2, u3, u4]`, где v1, v2 — виртуальные входы цепочек интеграторов.

#### Матрица управления b

```
b(φ, θ, ψ, u1) = Rz(φ) · B_inner
```

`B_inner` — внутренняя матрица аэродинамических усилий без перестановки строк (реализация в `drone_sim/control/path_following.py`).

#### Алгоритм согласованного управления (Гл. 4)

1. **Наблюдатель ближайшей точки** (Лемма 3):
   ```
   ζ̇ = −γ · sign(∂H/∂ζ) · H(ζ, x),   H = (p(ζ) − x)·t(ζ)
   ```
   Для прямых — аналитическая формула; для остальных кривых — итерационный наблюдатель `NearestPointObserver`.

2. **Ошибки в системе координат Френе** (ур. 60):
   ```
   λ̃₁ = [s_arc − V*t,  e₁,  e₂,  δφ]
   ```

3. **Наблюдатель производных** (ур. 73–77):
   Пятиступенчатый наблюдатель высокого усиления κ оценивает `λ̃₁` и её производные до 4-го порядка.

4. **Закон управления** (ур. 71–72):
   ```
   Ū = sat_L[b⁻¹ W⁻¹ (−σ − Σᵢ γᵢ λ̂ᵢ)]
   U = γ₅·η + Ū,   η̇ = Ū
   ```

#### Структура реализации в drone_sim

| Компонент | Модуль | Назначение |
|---|---|---|
| Физика | `models/dynamics.py` | `quad_dynamics_16` — правая часть ОДУ |
| Модель дрона | `models/quad_model.py` | `QuadModel` — параметры + ограничения |
| Кривые | `geometry/curves.py` | `CurveGeom`, `spiral_curve`, `line_xyz_curve` |
| Низкоуровневый регулятор | `control/path_following.py` | `Ch4PathController` — прямая/спираль |
| Высокоуровневый интерфейс | `simulation/path_sim.py` | `simulate_path_following` — произвольная кривая |
| Интегрирование | `simulation/integrators.py` | `rk4_step` — метод Рунге–Кутты 4-го порядка |

---

### ML-пайплайн: оптимизация V*

#### Постановка задачи

Задача: по вектору состояния дрона и геометрии текущей точки кривой предсказать оптимальную параметрическую скорость `V*`, максимизирующую скорость движения при сохранении стабильности слежения.

#### Входные признаки SpeedMLP (7 штук)

| # | Признак | Формула | Нормировка |
|---|---|---|---|
| 1 | `e1` | Тангенциальная ошибка в системе Френе | `tangential_error_limit` |
| 2 | `e2` | Поперечная ошибка в системе Френе | `lateral_error_limit` |
| 3 | `de2_dt` | Проекция скорости на нормаль кривой | `max_velocity_norm` |
| 4 | `v_norm` | Норма линейной скорости дрона | `max_speed` |
| 5 | `heading_error` | Угол между скоростью и касательной | π |
| 6 | `kappa` | Кривизна кривой в текущей точке | 1.0 |
| 7 | `kappa_max_lookahead` | Макс. кривизна на окне lookahead | 1.0 |

Реализация: `ml/dataset/features.py` — `feature_vector(state, curve, drone, s)`.

#### Архитектура SpeedMLP

```
Вход: 7 признаков
  → Linear(7 → 128) → ReLU
  → Linear(128 → 128) → ReLU
  → Linear(128 → 64) → ReLU
  → Linear(64 → 1)
  → Sigmoid × max_speed
Выход: V* ∈ [min_speed, max_speed]
```

Реализация: `ml/models/speed_model.py`.

#### Oracle: метка для обучения

Oracle перебирает V* в диапазоне `[min_speed, max_speed]` с шагом `speed_step` и для каждой кривой находит максимальное стабильное значение через короткий rollout-ролаут (20 с, kappa=100, dt=0.01). Стабильность определяется по трём критериям:
- `|e2| < lateral_error_limit`
- `||v|| < max_velocity_norm`
- отсутствие NaN/Inf в состоянии

Реализация: `ml/dataset/simulator_wrapper.py` — `find_optimal_speed`.

#### Пайплайн

```
Кривые (line/circle/spiral)
  → Oracle (rollout-симуляции)
  → dataset.csv (7 признаков + V_opt)
  → SpeedMLP (обучение с early stopping)
  → speed_model.pt (веса + параметры QuadModel)
  → SpeedPredictor.load() / SpeedPredictor.default()
  → speed_fn(state, s) → V*  для SimConfig.speed_fn
```

#### Параметры QuadModel для ML

Один `QuadModel` задаёт и физику дрона, и параметры для всего пайплайна:

```python
from drone_sim.models.quad_model import QuadModel

drone = QuadModel(
    max_speed=3.0,               # верхняя граница V* (oracle + нормировка v_norm)
    min_speed=0.3,               # нижняя граница V*
    lateral_error_limit=0.5,    # oracle: порог |e2|, нормировка e2
    tangential_error_limit=0.7, # нормировка e1
    max_velocity_norm=6.0,      # oracle: порог ||v||, нормировка de2_dt
)
```

Параметры сохраняются в `.pt`-чекпоинт и автоматически восстанавливаются при `SpeedPredictor.load()`.

---

## Использование пакета drone_sim

### Установка

```bash
pip install -e code/   # После этого drone_sim доступен из любой директории
```

### Полный пример

```python
import numpy as np
from drone_sim import make_curve, SimConfig, QuadModel, simulate_path_following

# Эллиптическая спираль
curve = make_curve(lambda s: np.array([4.0*np.cos(s), 2.0*np.sin(s), 0.5*s]))

x0 = np.zeros(16)
x0[0:3] = np.array([4.0, 0.0, 0.0])

cfg = SimConfig(
    quad_model=QuadModel(),
    Vstar=1.0, T=40.0, dt=0.002, x0=x0,
    kappa=200.0, gamma=(1., 3., 5., 3., 1.),
    gamma_nearest=5.0,
)

result = simulate_path_following(curve, cfg)
result.print_summary()
result.plot("code/out_images/my_curve")
```

### Ключевые параметры SimConfig

| Параметр | Описание | Рекомендуемое значение |
|---|---|---|
| `Vstar` | Параметрическая скорость (не дуговая!) | 1.0 |
| `T`, `dt` | Время и шаг RK4 | 30–40 с, 0.002 с |
| `kappa` | Коэффициент усиления наблюдателя | 200 при dt=0.002 |
| `gamma` | Коэффициенты регулятора (5-кортеж) | (1, 3, 5, 3, 1) |
| `gamma_nearest` | Коэффициент наблюдателя ближайшей точки | зависит от кривой |
| `nearest_fn` | Аналитика ближайшей точки | для прямых: `nearest_point_line` |
| `speed_fn` | NN-оптимизатор `callable(state, s) → V*` | `SpeedPredictor.load(...)` |

### Выходные данные SimResult

```python
result.t          # [n]: массив времени
result.x          # [n×16]: траектория состояния
result.errors     # [n×4]: s_arc−V*t, e1, e2, δφ
result.velocity   # [n]: норма скорости ||v||
result.zeta       # [n]: параметр ближайшей точки
result.p_ref      # [n×3]: ближайшие точки на кривой
```

---

## ML-пайплайн

### Полный запуск пайплайна

```bash
# Шаг 1. Собрать датасет (рекомендуемые параметры для полного обучения):
python code/scenarios/run_build_dataset.py --curves 200 --samples 10

# Шаг 2. Обучить SpeedMLP:
python code/scenarios/train_speed_model.py --epochs 200 --patience 20

# Шаг 3. Инференс (тестовый запуск с NN):
python code/scenarios/run_test_drone.py --model default

# Шаг 4. Сравнение с baseline:
python code/scenarios/run_nn_speed.py --curve spiral
```

### SpeedPredictor — глобальное имя модели

Модель по умолчанию доступна через:

```python
from ml.inference.predict import SpeedPredictor

# Загрузить из стандартного пути проекта (code/ml/data/model/speed_model.pt):
predictor = SpeedPredictor.default()

# Загрузить из конкретного файла:
predictor = SpeedPredictor.load("code/ml/data/model/speed_model.pt")

# Предсказать V* по вектору признаков:
V_star = predictor.predict(feature_vector(state, curve, drone=predictor.drone, s=zeta))
```

---

## Рекомендации и ограничения

### Допущения реализации

**Нормализованная физическая модель.** По умолчанию `QuadModel` использует `mass=1.0`, `J=1.0`. Регуляторные коэффициенты (`kappa`, `gamma`) нормированы под эту модель. При изменении физических параметров требуется перенастройка.

**Параметрическая vs дуговая скорость.** `Vstar` — скорость изменения параметра ζ, не дуговая скорость:
- Прямая `‖t‖=√3`: дуговая скорость ≈ 1.73 м/с при Vstar=1.0
- Спираль r=3, `‖t‖=√10`: дуговая скорость ≈ 3.16 м/с при Vstar=1.0

### Известные ограничения

**Зависимость kappa и dt.** Устойчивость наблюдателя производных требует выполнения:

| kappa | Максимальный dt | Применение |
|---|---|---|
| 100 | ≤ 0.010 с | Простые кривые, тестирование |
| 200 | ≤ 0.005 с | Стандарт диссертации (Гл. 4) |
| 300 | ≤ 0.002 с | Высокая точность или кривизна |

**Выбор gamma_nearest.** Условие устойчивости дискретного наблюдателя ближайшей точки:
```
0 < gamma_nearest × dt × ‖t‖²_max < 2
```

Рекомендуемые значения для стандартных кривых:
- Спираль r=3 (`‖t‖²=10`): `gamma_nearest=1`
- Спираль r=2 (`‖t‖²=5`): `gamma_nearest=3`
- Спираль r=1.5 (`‖t‖²=3.25`): `gamma_nearest=20`
- Эллипс `[4cos, 2sin, 0.5s]` (`‖t‖² ∈ [4.25, 16.25]`): `gamma_nearest=5`

**Прямые кривые.** `NearestPointObserver` нестабилен для прямых — необходима аналитическая `nearest_fn`:
```python
from drone_sim.geometry import nearest_point_line
cfg = SimConfig(..., nearest_fn=nearest_point_line)
```

**[ИЗВЕСТНАЯ ПРОБЛЕМА] Формула длины дуги для неравномерных кривых.**
В `simulation/path_sim.py` используется приближение `s_arc = ζ·‖t(ζ)‖`, точное только при `‖t‖=const`.
Для неравномерно параметризованных кривых (эллипс, парабола) это приводит к инверсии знака ошибки скорости и расходимости регулятора. Правильная формула — `s_arc = ∫₀^ζ ‖t(τ)‖ dτ`.
До исправления контроллер гарантированно работает только на кривых с `‖t‖=const`: прямые, круговые спирали.

**Начальное положение.** Дрон должен стартовать вблизи кривой (≲ 1–2 м). При большом начальном отклонении наблюдатель ближайшей точки может расходиться.

**ML-пайплайн.** Обучение SpeedMLP на малом датасете (< 500 записей) даёт нестабильные предсказания. Для воспроизводимых результатов рекомендуется `--curves 200 --samples 10` (≈ 2000 записей). Один и тот же `QuadModel` должен использоваться на всех трёх шагах: датасет → обучение → инференс.

**Коэффициенты Гл. 2 (архив).** Реализация использует K5=diag(8,8), K6=diag(2,2) вместо K5=diag(4,4), K6=diag(1,1) из диссертации. Причина: Python-модель без инерционных матриц J требует K5>6 для устойчивости при K4=6.

---

## Планируемые расширения

- **RL для V* (PPO/SAC).** Замена supervised oracle на обучение с подкреплением для адаптивного выбора V* без заранее собранного датасета. Placeholder: `drone_sim/nn/`.

- **Интеграция длины дуги.** Исправление формулы `s_arc = ∫₀^ζ ‖t(τ)‖ dτ` для поддержки произвольных неравномерно параметризованных кривых (эллипс, парабола, сплайн).

- **Расширение датасета.** Добавление новых типов кривых в генератор (сплайны, составные кривые, кривые с переменной кривизной).

- **Предобработка признаков.** Нормализация по статистике датасета (StandardScaler), добавление временны́х признаков (история ошибок).

- **Ансамблирование моделей.** Несколько независимо обученных SpeedMLP с усреднением или подбором по доверительному интервалу.
