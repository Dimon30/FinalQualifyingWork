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

5. **Сравнительный анализ** — сопоставление базовой симуляции с константной V* и нейросетевого оптимизатора по метрикам ошибки слежения (e1, e2, RMS) и средней скорости движения. Демонстрация: NN увеличивает скорость в 2.8× при сохранении устойчивости на спирали r=3.

6. **RL-архитектуры (offline) для V*** — реализованы три архитектуры (SAC, TD3, PPO), обучаемые на том же CSV-датасете, что и MLP, но с разными функциями потерь. Единый реестр моделей (`registry.py`) и `SpeedPredictorAny` обеспечивают взаимозаменяемый инференс.

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

# Движение вдоль горизонтального круга r=3, z=5 (||t||=1):
python code/scenarios/run_ch4_circle.py
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
# Базовая симуляция (спираль r=3, константная V*):
python code/scenarios/run_test_drone.py

# С NN-оптимизатором (требует обученной модели, см. шаги 6–7):
# --vstar-cap ограничивает V* сверху; для спирали r=3 граница устойчивости ~4.0, рекомендуется 3.5
python code/scenarios/run_test_drone.py --model auto --vstar-cap 3.5

# Указать собственную директорию для результатов:
python code/scenarios/run_test_drone.py --out code/out_images/my_experiment
```

Графики по умолчанию сохраняются в `code/out_images/test_drone/`.

### 5. Сборка небольшого датасета (тест)

```bash
# --oracle-horizon 4000 = 4000×0.005с = 20с — достаточно для проверки стабильности
# Без этого флага oracle использует горизонт ~1с (200 шагов × 0.005) и объявляет V*=9.9
# "стабильным" там, где реальная симуляция взрывается
python code/scenarios/run_build_dataset.py --curves 10 --samples 20 --oracle-horizon 4000
```

Датасет сохраняется в `code/ml/data/dataset.csv`.
Диагностические графики — в `code/out_images/dataset/`.

### 6. Обучение базовой MLP-модели

```bash
python code/scenarios/train_speed_model.py --epochs 100 --patience 15
```

Модель сохраняется в `code/ml/data/saved_models/speed_model.pt`.
Графики качества — в `code/out_images/training/`.

### 7. Инференс: симуляция с NN-оптимизатором

```bash
# Стандартный путь (code/ml/data/saved_models/speed_model.pt):
python code/scenarios/run_test_drone.py --model default --vstar-cap 3.5

# Авто-поиск последней модели в code/ml/data/:
python code/scenarios/run_test_drone.py --model auto --vstar-cap 3.5
```

> **Важно.** `--vstar-cap` ограничивает максимальную V*, выдаваемую нейросетью. Текущая модель может предсказывать V*≈9 во время полёта; без ограничения дрон уйдёт в расходимость. Рекомендуемое значение для спирали r=3: `--vstar-cap 3.5`.

### 8. Сравнение: константная V* vs MLP-оптимизатор

```bash
# На спирали r=3 (демо с проверенными параметрами):
# --vstar-cap 3.5    — ограничение V* сверху (граница устойчивости ~4.0)
# --vstar-rate 0.3   — максимальный темп изменения V* (с⁻¹), сглаживает скачки
python code/scenarios/run_nn_speed.py \
    --curve spiral \
    --model code/ml/data/saved_models/speed_model.pt \
    --vstar-cap 3.5 \
    --vstar-rate 0.3

# На прямой:
python code/scenarios/run_nn_speed.py --curve line --vstar-cap 3.5 --vstar-rate 0.3

# Сохранить в отдельную директорию:
python code/scenarios/run_nn_speed.py --curve spiral --vstar-cap 3.5 --out code/out_images/compare_spiral
```

Сравнительные графики сохраняются в `code/out_images/nn_speed/` (по умолчанию).

> **Результат демо** (спираль r=3): NN-оптимизатор увеличивает среднюю скорость в **2.8×** (1.0 → 2.8 м/с) при сохранении сходимости e₂ → 0, максимальная поперечная ошибка 0.043 м << предела 0.5 м.

### 9. Обучение RL-архитектуры (SAC / TD3 / PPO)

```bash
# Обучить SAC-модель (требует dataset.csv из шага 5):
python code/scenarios/train_rl_model.py --model sac --epochs 200 --patience 20

# TD3 и PPO аналогично:
python code/scenarios/train_rl_model.py --model td3 --epochs 400 --patience 40
python code/scenarios/train_rl_model.py --model ppo

# Модели сохраняются в code/ml/data/saved_models/{sac,td3,ppo}_model.pt
```

### 10. Сравнение RL-модели с константной V*

```bash
# По кодовому имени (автопоиск .pt файла):
python code/scenarios/run_compare_models.py --model sac --curve spiral --vstar-cap 3.5
python code/scenarios/run_compare_models.py --model td3 --curve line
python code/scenarios/run_compare_models.py --model ppo --curve circle

# Явный путь к чекпоинту:
python code/scenarios/run_compare_models.py \
    --model code/ml/data/saved_models/sac_model.pt --curve spiral --vstar-cap 3.5
```

Сравнительные графики сохраняются в `code/out_images/compare_rl/`.

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

  ml/                         — ML-пайплайн оптимизации V* (supervised + offline RL)
    config.py                 — MLConfig, OracleConfig, DEFAULT_MODEL_PATH
    curves/
      generator.py            — CurveSpec, make_line/circle/spiral_curve
    dataset/
      build_dataset.py        — generate_dataset: oracle + запись CSV
      features.py             — extract_features, feature_vector (7 признаков)
      simulator_wrapper.py    — rollout_with_speed, is_stable, find_optimal_speed
    models/
      speed_model.py          — SpeedMLP (128→128→64→1), save/load_speed_model
      sac_model.py            — SpeedSAC: Gaussian actor + twin Q-critics
      td3_model.py            — SpeedTD3: детерминированный актор + Polyak targets
      ppo_model.py            — SpeedPPO: Gaussian policy + value + clipped surrogate
      registry.py             — get_speed_model, save/load_speed_model_any, SpeedPredictorAny
    training/
      train_model.py          — train() с early stopping, TrainResult
      train_rl_models.py      — train_rl(): SAC/TD3/PPO offline на CSV
    inference/
      predict.py              — SpeedPredictor: load/predict/default
    data/                     — dataset.csv, saved_models/*.pt  ← не коммитить

  scenarios/
    run_ch4_line.py           — Гл. 4 диссертации: прямая (стр. 41)
    run_ch4_spiral.py         — Гл. 4 диссертации: спираль r=3 (стр. 43–44)
    run_ch4_circle.py         — горизонтальный круг r=3, z=5 (||t||=1)
    run_test_drone.py         — тестовый запуск дрона, спираль r=3 (поддержка --model, --vstar-cap)
    run_build_dataset.py      — сборка датасета (oracle V*)
    train_speed_model.py      — обучение SpeedMLP + графики качества
    run_nn_speed.py           — сравнение: константная V* vs MLP-оптимизатор
    train_rl_model.py         — обучение SAC/TD3/PPO/MLP на CSV (единый CLI)
    run_compare_models.py     — сравнение любой модели (mlp|sac|td3|ppo) с константной V*

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
   λ̃₁ = [s_arc − s_ref,  e₁,  e₂,  δφ]
   ```
   где `s_ref = ∫₀ᵗ V*(τ)dτ` — интегральный накопитель опорной скорости (при адаптивном V* корректнее, чем `V*·t`).


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

Oracle перебирает V* в диапазоне `[min_speed, max_speed]` с шагом `speed_step` и для каждой кривой находит максимальное стабильное значение через rollout-симуляцию (kappa=100, dt=0.005). Стабильность определяется по трём критериям:
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
    max_speed=10.0,              # верхняя граница V* (oracle + нормировка v_norm)
    min_speed=0.3,               # нижняя граница V*
    lateral_error_limit=0.5,    # oracle: порог |e2|, нормировка e2
    tangential_error_limit=0.7, # нормировка e1
    max_velocity_norm=10.0,     # oracle: порог ||v||, нормировка de2_dt
)
```

> **Важно.** `max_speed=10.0` должен быть одинаковым на всех трёх шагах (датасет → обучение → инференс). Несоответствие приводит к R²≈0 при обучении (метки ∈[0, 10], выход модели ∈[0, 3]).

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
| `warmup_time` | Время прогрева (speed_fn не активна) | 5.0 с |
| `vstar_max_rate` | Макс. темп изменения V* (с⁻¹), сглаживает скачки | 0.3 |

### Выходные данные SimResult

```python
result.t          # [n]: массив времени
result.x          # [n×16]: траектория состояния
result.errors     # [n×4]: s_arc−s_ref, e1, e2, δφ  (s_ref = ∫V*(τ)dτ — реальная ошибка контроллера)
result.velocity   # [n]: норма скорости ||v||
result.zeta       # [n]: параметр ближайшей точки
result.p_ref      # [n×3]: ближайшие точки на кривой
```

---

## ML-пайплайн

### Полный запуск пайплайна

```bash
# Шаг 1. Собрать датасет (рекомендуемые параметры для полного обучения):
# --oracle-horizon 4000 критично: 4000×0.005с = 20с (по умолчанию 200 шагов = 1с — недостаточно!)
# --coarse-fine — точнее (грубый шаг 0.5, затем точный 0.1), чуть медленнее
python code/scenarios/run_build_dataset.py \
    --curves 1000 --samples 10 \
    --oracle-horizon 4000

# Шаг 2. Обучить SpeedMLP:
python code/scenarios/train_speed_model.py --epochs 200 --patience 20

# Шаг 3. Инференс (тестовый запуск с NN):
python code/scenarios/run_test_drone.py --model default --vstar-cap 3.5

# Шаг 4. Сравнение с baseline:
python code/scenarios/run_nn_speed.py \
    --curve spiral \
    --model code/ml/data/saved_models/speed_model.pt \
    --vstar-cap 3.5 \
    --vstar-rate 0.3
```

### SpeedPredictor — глобальное имя модели

Модель по умолчанию доступна через:

```python
from ml.inference.predict import SpeedPredictor

# Загрузить из стандартного пути проекта (code/ml/data/saved_models/speed_model.pt):
predictor = SpeedPredictor.default()

# Загрузить из конкретного файла:
predictor = SpeedPredictor.load("code/ml/data/saved_models/speed_model.pt")

# Предсказать V* по вектору признаков:
V_star = predictor.predict(feature_vector(state, curve, drone=predictor.drone, s=zeta))
```

---

## RL-архитектуры (offline)

Реализованы три архитектуры, обучаемые на **том же CSV-датасете**, что и SpeedMLP, но с разными функциями потерь.

### Кодовые имена и архитектуры

| Имя | Класс | Описание |
|---|---|---|
| `mlp` | `SpeedMLP` | Полносвязный, MSE, supervised |
| `sac` | `SpeedSAC` | Gaussian actor + twin Q-critics, NLL + entropy |
| `td3` | `SpeedTD3` | Детерминированный актор, BC + Q-guided, Polyak targets |
| `ppo` | `SpeedPPO` | Gaussian policy + value, clipped surrogate + entropy |

Все архитектуры принимают 7 признаков и возвращают V* ∈ [min_speed, max_speed].

### Реестр моделей

```python
from ml.models.registry import get_speed_model, SpeedPredictorAny

# Создать необученную модель:
model = get_speed_model("sac", max_speed=10.0)

# Загрузить обученный предиктор из .pt файла:
pred = SpeedPredictorAny.load("code/ml/data/saved_models/sac_model.pt")
v = pred.predict(feature_vector(state, curve, drone=pred.drone, s=zeta))
```

Чекпоинт хранит поле `model_type` — `load_speed_model_any()` автоматически восстанавливает правильный класс. Файлы: `{mlp→speed_model.pt, sac→sac_model.pt, td3→td3_model.pt, ppo→ppo_model.pt}`.

### Обучение

```bash
# SAC (рекомендуется как первый кандидат):
python code/scenarios/train_rl_model.py --model sac --epochs 200 --patience 20

# TD3:
python code/scenarios/train_rl_model.py --model td3 --epochs 400 --patience 40

# PPO:
python code/scenarios/train_rl_model.py --model ppo

# MLP (аналог train_speed_model.py):
python code/scenarios/train_rl_model.py --model mlp
```

### Сравнение с константной V*

```bash
python code/scenarios/run_compare_models.py --model sac --curve spiral --vstar-cap 3.5 --vstar-rate 0.3
```

Выводит таблицу метрик (e2_final, e2_max, e2_rms, e1_rms, v_mean, v_final) и строит 4 графика в `code/out_images/compare_rl/`.

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

**[ИСПРАВЛЕНИЕ] Формула длины дуги для произвольных кривых.**
Прежнее приближение `s_arc = ζ·‖t(ζ)‖` было точным только при `‖t‖=const`.
Исправлено на инкрементальный интеграл `s_arc = ∫₀^ζ ‖t(τ)‖ dτ` (метод средней точки, обновляется в каждом шаге симуляции).
Эллиптическая спираль теперь работает: e₁=0.003 м, e₂=0.001 м.

**Начальное положение.** Дрон должен стартовать вблизи кривой (≲ 1–2 м). При большом начальном отклонении наблюдатель ближайшей точки может расходиться.

**ML-пайплайн.** Обучение SpeedMLP на малом датасете (< 500 записей) даёт нестабильные предсказания. Для воспроизводимых результатов рекомендуется `--curves 1000 --samples 10` (≈ 10 000 записей). Один и тот же `QuadModel` должен использоваться на всех трёх шагах: датасет → обучение → инференс.

[//]: # (**Коэффициенты Гл. 2 &#40;архив&#41;.** Реализация использует K5=diag&#40;8,8&#41;, K6=diag&#40;2,2&#41; вместо K5=diag&#40;4,4&#41;, K6=diag&#40;1,1&#41; из диссертации. Причина: Python-модель без инерционных матриц J требует K5>6 для устойчивости при K4=6.)

---

## Планируемые расширения

- **Обучение RL-моделей на большом датасете.** SAC/TD3/PPO реализованы, но обучение на датасете с `--oracle-horizon 4000` не проводилось. Ожидается более качественное предсказание V* по сравнению с MLP за счёт energy-based и Q-guided функций потерь.

- **Online RL для V*.** Замена offline обучения на средовую симуляцию: дрон получает reward за скорость при сохранении |e2| < порога. Placeholder: `drone_sim/nn/`.

- **Расширение датасета.** Добавление новых типов кривых (сплайны, составные кривые, переменная кривизна). Текущий датасет: line/circle/spiral.

- **Предобработка признаков.** Нормализация по статистике датасета (StandardScaler), добавление временны́х признаков (история ошибок).

- **Ансамблирование моделей.** Несколько независимо обученных моделей с усреднением или подбором по доверительному интервалу.
