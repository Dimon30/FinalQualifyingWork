# Адаптивное управление траекторным движением квадрокоптера

ВКР: «Исследование методов интеллектуального управления траекторным движением в трёхмерном пространстве для класса нелинейных систем».

Базовый регулятор: согласованное управление по выходу (Гл. 4 диссертации Ким С.А., 2024). Поверх него — четыре нейросетевые архитектуры (MLP / SAC / TD3 / PPO), адаптивно выбирающие параметрическую скорость V* на каждом шаге симуляции.

Активный код: `code_app/`. Архив Гл. 2–3: `legacy/`. Отчёт LaTeX: `report_app/`.

---

## Зависимости

```bash
pip install -e code_app/   # drone_sim + numpy + matplotlib
pip install pytest         # для тестов
pip install torch          # для ML/RL-моделей
pip install pandas         # для merge датасета
# или одной строкой
pip install -r requirements.txt
```

Минимум: Python 3.10+, numpy, matplotlib. Для ML — PyTorch.

---

## Воспроизведение за 30 минут (sanity check)

```bash
# 1. Установка
pip install -e code_app/ pytest

# 2. Тесты регулятора (6 кривых, ~10 с)
pytest code_app/tests/ -v

# 3. Базовые сценарии Гл. 4 (~30 с каждый)
python code_app/scenarios/run_ch4_line.py
python code_app/scenarios/run_ch4_spiral.py
python code_app/scenarios/run_ch4_circle.py

# 4. Эллиптическая спираль (~20 с)
python code_app/scenarios/run_test_drone.py
```

Все скрипты сохраняют графики в `code_app/out_images/`. Финальные ошибки e1, e2 должны быть < 0.01 м.

---

## Полное воспроизведение результатов ВКР (~24 часа)

### Шаг 1 — Генерация большого датасета (~18-20 ч, 8 терминалов параллельно)

```bash
# Терминал N (N от 1 до 8):
python code_app/scenarios/run_build_dataset.py \
  --curves 1000 --samples 12 \
  --oracle-horizon 4000 --oracle-speed-step 0.5 \
  --seed N --out code_app/ml/data/dataset_large_sN.csv --no-plots
```

Каждый процесс создаёт ~5,100 строк. Итого 8 файлов × ~5,100 = ~41,000 строк.

**Параметры обязательные**:
- `--oracle-horizon 4000` (20 с симуляции; меньше → ложные unstable-метки)
- `--no-plots` (отключает matplotlib в долгом цикле)
- `--seed N` (независимые случайные кривые → нет дублей)

### Шаг 2 — Слияние датасета (~1 мин)

```bash
python -c "
import pandas as pd, glob
parts = sorted(glob.glob('code_app/ml/data/dataset_large_s*.csv'))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df = df.sample(frac=1, random_state=0).reset_index(drop=True)
df.to_csv('code_app/ml/data/dataset_large.csv', index=False)
print(f'Merged {len(df)} rows from {len(parts)} files')
"
```

### Шаг 3 — Обучение моделей (~15 мин суммарно)

```bash
# MLP (5 мин)
python code_app/scenarios/train_speed_model.py \
  --csv code_app/ml/data/dataset_large.csv --epochs 300 --patience 30

# SAC (1 мин)
python code_app/scenarios/train_rl_model.py --model sac \
  --csv code_app/ml/data/dataset_large.csv --epochs 100 --patience 15 --lr 3e-4

# TD3 (1-2 мин)
python code_app/scenarios/train_rl_model.py --model td3 \
  --csv code_app/ml/data/dataset_large.csv --epochs 200 --patience 20 --lr 3e-4

# PPO (5 мин, чувствителен к lr — нужен lr ≤ 3e-4)
python code_app/scenarios/train_rl_model.py --model ppo \
  --csv code_app/ml/data/dataset_large.csv --epochs 300 --lr 3e-4 --batch 128
```

Чекпоинты сохранятся в `code_app/ml/data/saved_models/{speed,sac,td3,ppo}_model.pt`. Графики обучения — в `code_app/out_images/training/` (MLP) и `code_app/out_images/training_rl/` (SAC/TD3/PPO).

### Шаг 4 — Сравнение моделей на тестовой кривой (~3 мин)

```bash
for model in mlp sac td3 ppo; do
  python code_app/scenarios/run_compare_models.py \
    --model $model --curve spiral \
    --out report_app/images
done
```

Создаёт 16 PNG: `{model}_{errors,velocity,3d,sync}.png` в `report_app/images/`.

### Шаг 5 — Полный бенчмарк (~5 мин)

```bash
python code_app/scenarios/run_benchmark.py \
  --out code_app/out_images/benchmark \
  --report-images report_app/images
```

Запускает 4 модели на 4 кривых (spiral_r3, circle_r3z5, helix_r2, line_diag). В `report_app/images/` копируются 11 PNG + `summary_table.tex`.

### Шаг 6 — Графики датасета (~30 с)

```bash
python -c "
import sys, os
sys.path.insert(0, 'code_app')
os.chdir('code_app')
from scenarios.run_build_dataset import plot_dataset_stats, print_csv_summary
print_csv_summary('ml/data/dataset_large.csv')
plot_dataset_stats('ml/data/dataset_large.csv', '../report_app/images')
"
```

Создаёт 3 PNG: `dataset_vopt_kappa.png`, `dataset_features_scatter.png`, `dataset_correlations.png`.

### Шаг 7 — Сборка LaTeX-отчёта (~2 мин)

```bash
cd report_app
xelatex -interaction=nonstopmode report.tex && \
  biber report && \
  xelatex -interaction=nonstopmode report.tex && \
  xelatex -interaction=nonstopmode report.tex
```

Требования: `xelatex`, `biber`, шрифт Cambria (для русского текста). Итоговый PDF: `report_app/report.pdf` (~50 страниц).

---

## Структура проекта

```
code_app/
├── drone_sim/              — пакет: модели, геометрия, контроллер, симуляция
│   ├── models/             — quad_model.py, dynamics.py
│   ├── geometry/           — curves.py (CurveGeom, line/spiral)
│   ├── control/            — common.py, path_following.py
│   ├── simulation/         — path_sim.py (главный API), runner.py, integrators.py
│   └── visualization/      — plotting.py
├── ml/                     — ML-пайплайн оптимизации V*
│   ├── config.py           — MLConfig, OracleConfig
│   ├── curves/generator.py — генератор обучающих кривых (circle, spiral)
│   ├── dataset/            — build_dataset, features, simulator_wrapper
│   ├── models/             — speed_model, sac_model, td3_model, ppo_model, registry
│   ├── training/           — train_model.py, train_rl_models.py
│   ├── inference/          — predict.py (SpeedPredictor)
│   ├── evaluation/         — test_suite, benchmark, plots
│   └── data/               — dataset_large.csv, saved_models/*.pt
├── scenarios/              — точки входа (run_ch4_*, train_*, run_benchmark, ...)
└── tests/                  — pytest для 6 кривых

report_app/                 — LaTeX-исходники ВКР (XeLaTeX + biber)
├── chapter1.tex … chapter3.tex
├── images/                 — PNG для отчёта (создаётся в Шагах 4-6)
├── tables/                 — auto-generated (summary_table.tex)
└── report.tex              — главный файл

legacy/                     — архив симуляций Гл. 2-3 (не активно)
Диссертация на сайт.pdf     — математическая основа
CLAUDE.md                   — подробные технические заметки
```

---

## Тестирование

```bash
pytest code_app/tests/              # все 6 тестов
pytest code_app/tests/ -v           # с именами сценариев
pytest code_app/tests/ -k spiral_r3 # один тест
pytest code_app/tests/ --fast       # T × 0.25 (быстрый прогон)
```

Каждый тест запускает симуляцию вдоль кривой и проверяет ‖[e1, e2]‖_final < 1.5 м. На текущем коде все 6 проходят за ~10 с.

---

## Известные ограничения

- **TD3 нестабилен на helix_r2 и crowded-сценариях** при `vstar_max_rate=0.3`. Детерминированный актор без entropy-регуляризации экстраполирует на OOD-состояния (большая e2) в опасную сторону. На spiral_r3 / circle_r3z5 работает корректно.
- **PPO offline-обучение чувствительно к lr**: при lr=1e-3 политика коллапсирует (NaN). Использовать lr ≤ 3e-4 и batch ≥ 128.
- **Lines исключены из датасета** (`type_weights = [0.0, 0.45, 0.55]` в `build_dataset.py`): kappa=0 → V_opt = max_speed всегда, не даёт сигнала; oracle без `nearest_fn` всегда проваливается. Модель обобщается на прямые через поведение на кривых с малой кривизной.
- **Спирали ограничены β < 31°** (`k_max = 0.6r` в `curves/generator.py`): безопасный запас от физического предела β ≈ 49°.
- **Гл. 2 и Гл. 3**: в архиве `legacy/`. Гл. 2 использует K5=diag(8,8) вместо diag(4,4) из диссертации (нормализованная Python-модель без инерций). Гл. 3 — наблюдатель не успевает сойтись за 40 с при kappa=100, dt=0.01.

---

## Идеи для дополнительных экспериментов

1. **Ablation studies**: влияние `oracle_horizon` (2000/4000/6000), `oracle_speed_step` (0.3/0.5/1.0) и размера датасета (5k / 15k / 41k) на качество моделей.
2. **Feature importance** (SHAP / permutation): какие из 7 признаков (e1, e2, de2/dt, v_norm, heading_error, kappa, kappa_max_lookahead) действительно нужны для предсказания V*.
3. **Кросс-валидация по типу кривой**: train на circles, тест на spirals — проверка генерализации.
4. **Анализ V_opt(t) во время полёта**: визуализация adaptive V* в реальном времени + сопоставление с локальной кривизной.
5. **Inference cost**: время одного `predict()` для MLP / SAC / TD3 / PPO (важно для real-time on-board исполнения).
6. **Out-of-distribution тест**: эллипс, лемниската, трефл — модель никогда не видела таких кривых.
7. **Влияние `warmup_time` и `vstar_max_rate`**: при каком минимальном warmup и rate-limit модели остаются стабильны.
8. **Сравнение `kappa=100` vs `kappa=200`** в oracle: будет ли выше save rate при более стабильном наблюдателе.
9. **Lines с правильным `nearest_fn`**: добавить `nearest_fn` в `CurveSpec` и oracle, включить lines в датасет — улучшит ли модель прямые.
10. **Conservative inference для SAC/TD3**: при предсказании далеко от обучающего распределения — клипить V* по статистикам датасета (например, mean + 2·std).

---

## Документация

- `CLAUDE.md` — детальные технические заметки (oracle-фиксы, safety monitor, особенности реализации).
- `Диссертация на сайт.pdf` — математическая основа (уравнения 56-77, Лемма 3).
- `report_app/report.tex` — итоговый отчёт ВКР.
