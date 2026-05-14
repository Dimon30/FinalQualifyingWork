# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## О проекте

Выпускная квалификационная работа по теме **«Исследование методов интеллектуального управления траекторным движением в трёхмерном пространстве для класса нелинейных систем»**. Объект управления — квадрокоптер (БПЛА).

Из диссертации Ким С.А. (2024) реализованы три вида регуляторов:
- Гл. 2 — управление по состоянию (архив: `legacy/code_ch2_ch3_04_13_archive/`)
- Гл. 3 — следящее управление по выходу (архив: `legacy/code_ch2_ch3_04_13_archive/`)
- **Гл. 4 — согласованное управление по выходу** (`code/`) — **активная разработка**

Исходная диссертация: `Диссертация на сайт.pdf` (в корне).
LaTeX-отчёт: `report/` (основной), `report_practice/` (отчёт по практике).

**Нейросетевой оптимизатор V***: реализованы четыре архитектуры в `code/ml/`:
- **MLP (SpeedMLP)** — supervised регрессия по oracle-меткам
- **SAC (SpeedSAC)** — стохастический актор + twin Q-critics + энтропийный бонус
- **TD3 (SpeedTD3)** — детерминированный актор + Polyak-таргеты + delayed update
- **PPO (SpeedPPO)** — clipped surrogate + value head + энтропийный бонус

Все 4 модели обучаются на одном CSV-датасете (offline), предсказывают скалярную V*.
Единый реестр `ml/models/registry.py` → `SpeedPredictorAny` — взаимозаменяемый инференс.

## Запуск симуляций

Все команды выполняются из **корня проекта**:

```bash
# Сценарии Гл. 4 (согласованное управление):
python code_app/scenarios/run_ch4_line.py     # прямая
python code_app/scenarios/run_ch4_spiral.py   # спираль r=3
python code_app/scenarios/run_ch4_circle.py   # круг r=3, z=5

# Тест эллиптической спирали:
python code_app/scenarios/run_test_drone.py                      # константная V*
python code_app/scenarios/run_test_drone.py --model auto         # NN-оптимизатор (авто)

# ML-пайплайн оптимизации V*:
python code_app/scenarios/run_build_dataset.py --curves 200 --samples 10 --oracle-horizon 4000
python code_app/scenarios/train_speed_model.py --epochs 200 --patience 20
python code_app/scenarios/run_nn_speed.py --curve spiral --vstar-rate 0.3

# RL-архитектуры (offline):
python code_app/scenarios/train_rl_model.py --model sac --epochs 200 --patience 20
python code_app/scenarios/train_rl_model.py --model td3 --epochs 400 --patience 40
python code_app/scenarios/train_rl_model.py --model ppo

# Запуск отдельной модели на тестовой кривой:
python code_app/scenarios/run_compare_models.py --model sac --curve spiral --vstar-rate 0.3
python code_app/scenarios/run_compare_models.py --model td3 --curve line
python code_app/scenarios/run_compare_models.py --model ppo --curve circle

# Полный бенчмарк (все 4 модели × 4 кривые):
python code_app/scenarios/run_benchmark.py
python code_app/scenarios/run_benchmark.py --models mlp,sac --curves spiral_r3,circle_r3z5
python code_app/scenarios/run_benchmark.py --out code_app/out_images/benchmark --report_app-images report_app/images

# Тесты (pytest):
pytest code_app/tests/                         # все 6 тестов
pytest code_app/tests/ -v                      # с именами
pytest code_app/tests/ -k spiral_r3           # конкретный тест
pytest code_app/tests/ --fast                  # ускоренный прогон (T×0.25)
```

Результаты (графики `.png`) сохраняются в `code/out_images/` (создаётся автоматически).

> Симуляции Гл. 2 и Гл. 3 перемещены в архив `legacy/code_ch2_ch3_04_13_archive/` и из активной разработки исключены.

## Сборка LaTeX-отчёта

```bash
cd "Z:/ITMO/FinalQualifyingWork/report"
xelatex -interaction=nonstopmode report_app.tex && biber report_app && xelatex -interaction=nonstopmode report_app.tex && xelatex -interaction=nonstopmode report_app.tex
```

Движок: **XeLaTeX** (fontspec, Cambria). Класс: `extreport` 14pt (пакет extsizes).
Библиография: `references.bib` + biber. Текущий объём: ~33 страницы.

## Зависимости

```bash
pip install -e code_app/   # устанавливает drone_sim + numpy + matplotlib (рекомендуется)
pip install pytest     # для запуска тестов
pip install torch      # для ML-пайплайна (SpeedMLP + RL-модели)
# или
pip install -r requirements.txt
```

## Архитектура

### Структура директорий

```
code/
  drone_sim/              — основной Python-пакет
    __init__.py           — публичный API: make_curve, SimConfig, simulate_path_following, …
    models/
      quad_model.py       — QuadModel (физические + лётные параметры: g, mass, J_*, max_speed, …)
      dynamics.py         — quad_dynamics_16, quad_dynamics_12, sat_tanh, G
    geometry/
      curves.py           — CurveGeom, line_xyz_curve, spiral_curve, nearest_point_line, se_from_pose
    control/
      common.py           — HighGainParams, DerivativeObserver4
      path_following.py   — Ch4PathController, W_mat, W_inv, b_mat, _safe_inv4
    simulation/
      integrators.py      — rk4_step
      runner.py           — simulate (основной цикл RK4)
      path_sim.py         — NearestPointObserver, PathFollowingController, SimConfig, SimResult, simulate_path_following
    visualization/
      plotting.py         — plot_3d_traj, plot_errors, plot_velocity, plot_angles, plot_xy, display_path, ensure_out
    nn/
      __init__.py         — placeholder (зарезервировано)
  ml/                     — ML-пайплайн оптимизации V* (supervised + offline RL)
    config.py             — MLConfig, OracleConfig, константы
    curves/
      generator.py        — CurveSpec, make_line/circle/spiral_curve, generate_dataset_curves
    dataset/
      build_dataset.py    — generate_dataset: oracle + запись CSV
      features.py         — extract_features, feature_vector (7 признаков)
      simulator_wrapper.py— rollout_with_speed, is_stable, find_optimal_speed
    models/
      speed_model.py      — SpeedMLP (128→128→64→1), save/load_speed_model
      sac_model.py        — SpeedSAC: Gaussian actor + twin Q-critics
      td3_model.py        — SpeedTD3: детерминированный актор + Polyak targets
      ppo_model.py        — SpeedPPO: Gaussian policy + value head + clipped surrogate
      registry.py         — get_speed_model, save/load_speed_model_any, SpeedPredictorAny
    training/
      train_model.py      — train() с early stopping, TrainResult (для MLP)
      train_rl_models.py  — train_rl(): SAC/TD3/PPO offline на CSV
    inference/
      predict.py          — SpeedPredictor (load/predict, восстановление QuadModel из .pt)
    evaluation/           — модуль автоматизированной оценки
      test_suite.py       — TestScenario, get_test_suite() → 4 фиксированных сценария
      benchmark.py        — BenchmarkRunner, ModelResult
      plots.py            — plot_e2/velocity_comparison, plot_summary_bar, save_latex_table
    data/                 — dataset.csv, saved_models/*.pt (не коммитить)
  scenarios/
    run_ch4_line.py       — сценарий из диссертации: прямая
    run_ch4_spiral.py     — сценарий из диссертации: спираль r=3
    run_ch4_circle.py     — горизонтальный круг r=3, z=5 (||t||=1)
    run_test_drone.py     — эллипт. спираль с поддержкой --model (NN или константная V*)
    run_build_dataset.py  — сборка датасета (oracle V*)
    train_speed_model.py  — обучение SpeedMLP + графики качества
    run_nn_speed.py       — сравнение: baseline константная V* vs MLP-оптимизатор
    train_rl_model.py     — обучение SAC/TD3/PPO/MLP на CSV (единый CLI)
    run_compare_models.py — одна модель vs baseline на одной кривой
    run_benchmark.py      — полный бенчмарк: все модели × все тестовые кривые
  tests/
    conftest.py           — добавляет code/ в sys.path + --fast hook
    test_curves.py        — pytest: 6 кривых, PASS/FAIL по ||[e1,e2]|| < 1.5 м
  conftest.py             — добавляет code/ в sys.path (для запуска pytest из корня)
  pytest.ini              — testpaths = tests, pythonpath = .
code/out_images/          — сгенерированные графики (не коммитить)
  benchmark/              — результаты run_benchmark.py (e2/velocity по сценариям + summary)
legacy/                   — архив симуляций Гл. 2–3 (не трогать)
report/                   — основной LaTeX-отчёт ВКР
  images/                 — изображения для report/ (benchmark + индивидуальные модели)
  tables/                 — auto-generated таблицы (summary_table.tex)
report_practice/          — отчёт по практике
Диссертация на сайт.pdf   — математическая основа
```

### Вектор состояния (16-мерный, Гл. 3–4)

```
x = [x, y, z,  vx, vy, vz,  φ, θ, ψ,  φ̇, θ̇, ψ̇,  u1_bar, ρ1, u2, ρ2]
```

- **Соглашение об углах**: φ = рысканье (yaw), θ = тангаж (pitch), ψ = крен (roll) — **нестандартный порядок!**
- `u1_bar, ρ1` — состояние двойного интегратора тяги; реальная тяга `u1 = sat_tanh(u1_bar, L)`
- `u2, ρ2` — состояние двойного интегратора по рысканью
- Управляющий вход `U = [v1, v2, u3, u4]`: v1, v2 — виртуальные входы цепочек интеграторов; u3 = θ̈, u4 = ψ̈

Для Главы 2 используется 12-мерный вектор без интеграторных расширений.

### Публичный API (`drone_sim`)

```python
from drone_sim import make_curve, SimConfig, simulate_path_following

SimConfig(
    Vstar=1.0,           # параметрическая скорость (не дуговая!)
    T=30.0, dt=0.002,    # время и шаг RK4
    kappa=200.0,         # kappa=100 если dt>0.005
    gamma=(1,3,5,3,1),   # γ1..γ4 + γ5 (eta-интегратор)
    nearest_fn=None,     # аналитика для прямой: nearest_point_line
    quad_model=None,     # None = нормализованная модель (mass=1)
    speed_fn=None,       # callable(state, s) → V* для NN-режима
    warmup_time=5.0,     # секунд до включения speed_fn
    vstar_max_rate=0.3,  # макс. темп изменения V* (с⁻¹)
)

result.errors   # [n×4]: [s_arc−s_ref, e1, e2, delta_phi]
result.velocity # [n]: ||v|| дуговая скорость
result.x        # [n×16]: полный вектор состояния
result.zeta     # [n]: параметр ближайшей точки
result.print_summary()
result.plot("out_images/dir")
```

### QuadModel — лётные ограничения

```python
from drone_sim.models.quad_model import QuadModel

drone = QuadModel(
    g=9.81, mass=1.0, J_phi=1.0, J_theta=1.0, J_psi=1.0,
    max_speed=10.0,              # верхняя граница V* для oracle и нормировки v_norm
    min_speed=0.3,               # нижняя граница V*
    max_velocity_norm=10.0,      # порог ||v||: превышение = нестабильность
    lateral_error_limit=0.5,     # порог |e2|: oracle критерий стабильности
    tangential_error_limit=0.7,  # порог |e1|: нормировка признака
    nan_is_failure=True,
)
```

Параметры `QuadModel` сохраняются в `.pt`-чекпоинт и восстанавливаются при `SpeedPredictorAny.load()`.
**Важно**: `max_speed=10.0` должен совпадать на всех трёх шагах: датасет → обучение → инференс.

### ML-реестр и инференс

```python
from ml.models.registry import SpeedPredictorAny

# Загрузить любую модель (тип определяется автоматически из чекпоинта):
pred = SpeedPredictorAny.load("code_app/ml/data/saved_models/sac_model.pt")
V_star = pred.predict(feature_vector(state, curve, drone=pred.drone, s=zeta))
```

Чекпоинты: `{speed,sac,td3,ppo}_model.pt` в `code/ml/data/saved_models/`.

### Слои системы управления (Гл. 4)

- `Ch4PathController` (`drone_sim.control.path_following`) — низкоуровневый регулятор (ур. 71–77), только спираль и прямая.
- `PathFollowingController` (`drone_sim.simulation.path_sim`) — **рекомендуемый класс**, произвольная кривая.

Ключевые детали реализации:
- Ошибки в системе Френе: `λ̃₁ = [s_arc − s_ref, e₁, e₂, Δφ]`, матрица поворота `W(α, β, ε)`.
- `s_ref = ∫₀ᵗ V*(τ)dτ` — интегральный накопитель (НЕ `V*·t`), обновляется в конце `step()`.
- Safety monitor: при `|e2| > 0.5·lateral_limit` или `s_arc−s_ref < −0.3` → V* только уменьшается.
- Pre-loop reset: после первого (pre-loop) вызова `step()` при t=0 состояние сбрасывается.
- Матрица `b(φ, θ, ψ, u1) = Rz(φ) · B_inner` (без перестановки строк).

### Ключевые параметры (`HighGainParams`)

- `kappa` — усиление наблюдателя: `kappa=100 → dt≤0.01`, `kappa=200 → dt≤0.005`.
- `gamma` — коэффициенты регулятора (5-кортеж; γ₅ — вес η-интегратора).
- `gamma_nearest` — наблюдатель ближайшей точки: `γ < 2/(‖t‖²_max · dt)`.

## Особенности и известные отличия от диссертации

**Коэффициенты Гл. 2**:
- Диссертация: K5=diag(4,4), K6=diag(1,1). Python-код: K5=diag(8,8), K6=diag(2,2).
- Причина: Python-модель не имеет инерционных матриц J — условие устойчивости `K5·K6 > K4` требует K5>6 при K4=6.

**Скорость в Гл. 4 — дуговая vs. параметрическая**:
- `Vstar=1.0` — скорость изменения параметра ζ, не дуговая скорость.
- Длина дуги вычисляется инкрементально: `s_arc = ∫₀^ζ ‖t(τ)‖ dτ` (метод средней точки).
- Для прямой ‖t‖=√3 → V≈1.73 м/с; для спирали r=3 ‖t‖=√10 → V≈3.16 м/с.

**Наблюдатель в Гл. 3 — порядок обновления** (исправлено): (1) `hat()`, (2) вычислить U, (3) `step(y, y4_model=b4@U)`.

## Инструкции для ИИ-ассистента

- **Язык общения**: всегда отвечать на русском языке.
- **Среда разработки**: Windows, запуск через bash из корня проекта.
- **Предметная область**: ТАУ — робастное управление, нейронные сети (полносвязные), offline RL (SAC/TD3/PPO).
- **Диссертация**: при вопросах о математике — сначала свериться с `Диссертация на сайт.pdf`.
- **b-матрица**: `b = Rz(φ) @ B_inner` без перестановки строк → `drone_sim/control/path_following.py`.
- **Новый код** добавлять в `drone_sim/` (нейросети — в `drone_sim/nn/`), новые сценарии — в `scenarios/`, новые тесты — в `tests/`.
- **Импорты**: использовать `from drone_sim import ...` или `from drone_sim.submodule import ...`. Не создавать новые flat-файлы в `code/`.
- **Вывод в терминал**: если вывод некорректный для русских символов — меняй кодировку сразу.

## Текущий статус

### Реализовано (архив: `legacy/code_ch2_ch3_04_13_archive/`)

- **Гл. 2 (пример 1)**: стабилизация в точке x*=(1, 0.5, 2) — ||e||=0.0000 м ✓
- **Гл. 2 (пример 2)**: движение по ломаной A→B→C→D(30,20,20), T=30с, ||e||~2.07 м ✓
- **Гл. 3**: следящее управление, спираль r=0.5, T=40с — ||e||~0.50 м ✓

### Реализовано (активный код `code/`)

- **Гл. 4 прямая** (`scenarios/run_ch4_line.py`): e1,e2 → 0, s_arc−V*t → 0, ~1.73 м/с ✓
- **Гл. 4 спираль** (`scenarios/run_ch4_spiral.py`): e1,e2 → 0, s_arc−V*t → 0, ~3.16 м/с ✓
- **Гл. 4 круг** (`scenarios/run_ch4_circle.py`): горизонтальный круг r=3, z=5, ||t||=1 ✓
- **Тест дрона** (`scenarios/run_test_drone.py`): эллипт. спираль, опциональный NN-режим ✓
- **Пакет** `drone_sim`: полная модульная архитектура ✓
- **Тесты** (`tests/test_curves.py`): 6 кривых, pytest, `--fast` режим ✓

### Реализовано (`code/ml/`)

- **4 архитектуры V*-оптимизатора**: MLP, SAC, TD3, PPO — обучены и протестированы ✓
- **Единый реестр** `registry.py` + `SpeedPredictorAny` — взаимозаменяемый инференс ✓
- **Модуль оценки** `ml/evaluation/` — BenchmarkRunner, test_suite, plots ✓
- **Полный бенчмарк** (4 модели × 4 кривые, 2026-04-21):

| Модель | spiral_r3 | circle_r3z5 | helix_r2 | line_diag |
|---|---|---|---|---|
| MLP | 1.34× ✓ | 1.34× ✓ | 1.34× ✓ | 1.31× ✓ |
| SAC | 2.52× ✓ | 2.54× ✓ | 2.45× ✓ | 2.44× ✓ |
| TD3 | 2.81× ✓ | 2.81× ✓ | DIVERGED | 2.87× |
| PPO | 2.81× ✓ | 2.81× ✓ | 3.15× ✓ | 3.38× ✓ |

- **Oracle horizon**: критично использовать `--oracle-horizon 4000` (дефолт слишком короткий).
- **Safety monitor** в `path_sim.py`: NaN-fallback + e2-backoff + sarc-backoff.
- **s_ref интегратор**: `s_ref = ∫₀ᵗ V*(τ)dτ` (НЕ `V*·t`), обновляется В КОНЦЕ `step()`.
- **Pre-loop reset**: runner.py вызывает `step()` дважды при t=0; после первого вызова — сброс.

### Проблемы и ограничения

- **Гл. 2 — коэффициенты**: K5, K6 отличаются от диссертации (см. выше).
- **Гл. 3 — ошибка ~0.50 м**: наблюдатель не успевает сойтись за T=40с при kappa=100, dt=0.01.
- **[ИСПРАВЛЕНО] Гл. 4 — формула длины дуги**: инкрементальный интеграл `∫‖t(τ)‖dτ`. Эллипс: e1=0.003, e2=0.001 м ✓
- **[ИСПРАВЛЕНО] report/report.tex**: был `extrarcticle` → исправлен на `extreport` (14pt + главы).
- **TD3 нестабилен на helix_r2**: расходится при cap=4.0 (e2=20м). На остальных кривых работает.
- **PPO offline R²≈−0.98**: коллапс к константе ~5.2 при обучении, но в замкнутом контуре работает за счёт rate-limiter.
