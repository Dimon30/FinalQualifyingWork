# ML-пайплайн: адаптивное управление скоростью V\*

Модуль `ml/` реализует нейросетевую замену константной желаемой скорости V\*
в регуляторе Гл. 4. Вместо фиксированного числа контроллер получает
предсказание оптимальной скорости из обученного MLP на каждом шаге.

```
Симулятор → Oracle → CSV-датасет → обучение MLP → SpeedPredictor → PathFollowingController
```

---

## Быстрый старт

```bash
# 1. Сгенерировать датасет (10 кривых × 5 точек = ~50 записей)
python -m ml.dataset.build_dataset --num-curves 10

# 2. Обучить модель
python -m ml.training.train_model --csv code/ml/data/dataset.csv

# 3. Запустить симуляцию с NN-скоростью
python code/scenarios/run_ch4_spiral.py   # пример — нужно подключить predictor вручную
```

Все команды выполняются из **корня проекта** (не из `code/`).

---

## Структура

```
ml/
  config.py                   — OracleConfig, константы пайплайна
  dataset/
    curve_generator.py        — фабрики и валидация кривых для датасета
    simulator_wrapper.py      — rollout, is_stable, find_optimal_speed (oracle)
    features.py               — extract_features, feature_vector (7 признаков)
    build_dataset.py          — сборка CSV, CLI
  models/
    speed_model.py            — SpeedMLP, save_speed_model, load_speed_model
  training/
    train_model.py            — train(), _EarlyStopping, CLI
  inference/
    predict.py                — SpeedPredictor (load / predict / save)
  curves/
    generator.py              — CurveSpec, generate_curve
  data/                       — CSV, .pt-файлы (создаётся автоматически)
```

---

## Шаг 1 — Сборка датасета

### Что делает oracle

Для каждой пары `(состояние дрона, кривая)` oracle перебирает скорости
`V ∈ [min_speed, max_speed]` и выбирает **максимальную стабильную** V\*.
Стабильность определяется коротким ролаутом (по умолчанию 30 шагов RK4):
если `max |e2| < drone.lateral_error_limit` и нет NaN/разлёта — значение принято.

### CLI

```bash
python -m ml.dataset.build_dataset \
    --num-curves 200 \          # число кривых (не все пройдут валидацию)
    --samples 5 \               # точек на кривую
    --out code/ml/data/dataset.csv \
    --seed 42 \
    --coarse-fine               # двухпроходный поиск 0.5 → 0.1 (точнее, медленнее)
```

### Python API

```python
from ml.dataset.build_dataset import generate_dataset
from ml.config import OracleConfig
from drone_sim.models.quad_model import QuadModel

drone = QuadModel(max_speed=3.0, min_speed=0.3, lateral_error_limit=0.5)
oracle = OracleConfig(rollout_horizon=30, speed_step=0.3)

path = generate_dataset(
    num_curves=200,
    out_path="code/ml/data/dataset.csv",
    seed=42,
    n_samples_per_curve=5,
    drone=drone,
    oracle_cfg=oracle,
    coarse_to_fine=False,
)
print("CSV сохранён:", path)
```

### Формат CSV

| Колонка              | Описание                                  | Входит в MLP |
|----------------------|-------------------------------------------|:------------:|
| `e1`                 | тангенциальная ошибка (норм.)             | ✓            |
| `e2`                 | поперечная ошибка (норм.)                 | ✓            |
| `de2_dt`             | производная e2 (норм.)                    | ✓            |
| `v_norm`             | скорость дрона / max_speed                | ✓            |
| `heading_error`      | угол между v и касательной [0, 1]         | ✓            |
| `kappa`              | кривизна в текущей точке (норм.)          | ✓            |
| `kappa_max_lookahead`| макс. кривизна на окне вперёд (норм.)     | ✓            |
| `s`                  | параметр точки на кривой (диагностика)    | —            |
| `t_norm`             | `‖t(s)‖` (диагностика)                   | —            |
| `V_opt`              | целевая переменная V\*                    | —            |

### Ограничение: допустимые кривые

Контроллер Гл. 4 корректно работает **только** при `‖t(s)‖ = const`.
`validate_curve` автоматически отсеивает кривые с переменной нормой.
Поддерживаемые типы:

| Тип     | Формула `p(s)`                          | `‖t‖`            |
|---------|-----------------------------------------|------------------|
| прямая  | `[a·s, b·s, c·s]`                       | `√(a²+b²+c²)`    |
| окружность | `[r·cos(s/r), r·sin(s/r), 0]`        | `1`              |
| спираль | `[r·cos(s), r·sin(s), k·s]`             | `√(r²+k²)`       |

---

## Шаг 2 — Обучение модели

### CLI

```bash
python -m ml.training.train_model \
    --csv code/ml/data/dataset.csv \
    --out code/ml/data/model/speed_model.pt \
    --epochs 200 \
    --batch 64 \
    --lr 1e-3 \
    --patience 20 \       # early stopping: эпох без улучшения val_loss
    --val-frac 0.2 \
    --max-speed 3.0        # должно совпадать с drone.max_speed
```

### Python API

```python
from ml.training.train_model import train

result = train(
    csv_path="code/ml/data/dataset.csv",
    model_path="code/ml/data/model/speed_model.pt",
    max_speed=3.0,
    n_epochs=200,
    batch_size=64,
    lr=1e-3,
    patience=20,
)
print(f"Best val MSE: {result.best_val_loss:.6f}")
print(f"Stopped at epoch: {result.stopped_epoch}")
```

### Архитектура SpeedMLP

```
Linear(7 → 128) → ReLU
Linear(128 → 128) → ReLU
Linear(128 → 64)  → ReLU
Linear(64 → 1)    → sigmoid × max_speed
```

Выход: `V_pred ∈ (0, max_speed]`. Никогда не превышает физический предел дрона.
Итого параметров: ~25 857.

---

## Шаг 3 — Инференс

### Загрузка и предсказание

```python
from ml.inference.predict import SpeedPredictor
from ml.dataset.features import feature_vector
from drone_sim.models.quad_model import QuadModel

drone = QuadModel(max_speed=3.0, min_speed=0.3)
predictor = SpeedPredictor.load("code/ml/data/model/speed_model.pt", drone=drone)

# feat — вектор из 7 признаков для текущего состояния
feat = feature_vector(state, curve, drone=drone, s=s)
V = predictor.predict(feat)   # float, гарантированно в [min_speed, max_speed]
```

`SpeedPredictor.predict` всегда работает на **CPU** и применяет `clip` по
`[drone.min_speed, drone.max_speed]` поверх sigmoid.

### Подключение к симуляции через speed_fn

`SimConfig` принимает параметр `speed_fn: Callable[[state, s], float]`.
Контроллер вызывает его на каждом шаге и ограничивает изменение скорости
через `max_accel * dt` (rate limiting). При исключении — автоматический fallback
на предыдущее значение V.

```python
from drone_sim import make_curve, SimConfig, simulate_path_following
from ml.inference.predict import SpeedPredictor
from ml.dataset.features import feature_vector

predictor = SpeedPredictor.load("code/ml/data/model/speed_model.pt")
curve = make_curve(lambda s: ...)

def speed_fn(state, s):
    feat = feature_vector(state, curve, s=s)
    return predictor.predict(feat)

cfg = SimConfig(Vstar=1.0, T=30.0, dt=0.002, kappa=200, speed_fn=speed_fn)
result = simulate_path_following(curve, cfg)
result.print_summary()
```

---

## Конфигурация

### OracleConfig (ml/config.py)

| Параметр           | Умолчание | Описание                                   |
|--------------------|-----------|--------------------------------------------|
| `rollout_horizon`  | `30`      | шагов RK4 на один проверочный ролаут        |
| `speed_step`       | `0.3`     | шаг перебора в линейном режиме             |
| `coarse_step`      | `0.5`     | шаг первого прохода (coarse-to-fine)       |
| `fine_step`        | `0.1`     | шаг второго прохода (coarse-to-fine)       |
| `min_stable_steps` | `10`      | зарезервировано                            |

### QuadModel (drone_sim/models/quad_model.py)

Все пороги стабильности — в объекте дрона, не константы в коде:

| Параметр                | Умолчание | Описание                                  |
|-------------------------|-----------|-------------------------------------------|
| `max_speed`             | `3.0`     | V\* не превысит это значение              |
| `min_speed`             | `0.3`     | нижняя граница clip                       |
| `max_accel`             | `1.0`     | лимит `dV/dt` в контроллере               |
| `lateral_error_limit`   | `0.5`     | порог `max_e2` для критерия стабильности  |
| `tangential_error_limit`| `0.7`     | масштаб нормировки `e1`                   |
| `max_velocity_norm`     | `6.0`     | порог разлёта `‖v‖`                       |
| `nan_is_failure`        | `True`    | NaN/Inf → нестабильно                     |

---

## Расширение: добавление новой архитектуры модели

1. **Создать модель** в `ml/models/my_model.py`:

```python
import torch.nn as nn

class MySpeedNet(nn.Module):
    def __init__(self, max_speed=3.0):
        super().__init__()
        self.max_speed = max_speed
        self.net = nn.Sequential(
            nn.Linear(7, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)) * self.max_speed
```

2. **Добавить save/load функции** по образцу `speed_model.py`:
   `save_my_model(model, path)` / `load_my_model(path)`.

3. **Обучить** через `train_model.py` либо собственным циклом — формат CSV и
   имена колонок фиксированы, менять не нужно.

4. **Обернуть** в `SpeedPredictor` или аналогичный класс с методом `predict(feat) -> float`.

5. **Подключить** через `speed_fn` в `SimConfig` — контроллер не зависит от
   конкретной архитектуры модели.

> **Важно**: входной вектор признаков всегда имеет размер 7.  
> Порядок: `[e1, e2, de2_dt, v_norm, heading_error, kappa, kappa_max_lookahead]`.  
> Источник: `feature_vector()` из `ml/dataset/features.py`.

---

## Расширение: добавление нового типа кривой

1. Проверить, что для кривой выполняется `‖t(s)‖ = const` (аналитически или через `validate_curve`).

2. Добавить фабрику в `ml/dataset/curve_generator.py`:

```python
def make_my_curve(param: float) -> Curve:
    """Описание кривой. ||t|| = <значение>."""
    def p(s: float) -> np.ndarray:
        return np.array([...])
    return p
```

3. Зарегистрировать в `ml/curves/generator.py` в функции `generate_curve` —
   по аналогии с `"line"`, `"circle"`, `"spiral"`.

4. Перегенерировать датасет: `--num-curves` теперь будет включать новый тип.

---

## Диагностика

### Проверка валидности кривой

```python
from ml.dataset.curve_generator import validate_curve, make_spiral

ok = validate_curve(make_spiral(r=2.0, k=1.0), s_range=(0, 15))
print("Кривая допустима:", ok)
```

### Ролаут вручную

```python
from ml.dataset.simulator_wrapper import rollout_with_speed, is_stable
import numpy as np

state = np.zeros(16); state[12] = 9.81
metrics = rollout_with_speed(state, curve, V=1.5, horizon=50, zeta0=0.0)
print(metrics)         # {'max_e2': ..., 'final_e2': ..., 'nan_detected': ..., ...}
print(is_stable(metrics))
```

### Отдельное предсказание признаков

```python
from ml.dataset.features import extract_features

feats = extract_features(state, curve, s=3.0)
for k, v in feats.items():
    print(f"  {k:25s} = {v:+.4f}")
```

---

## Известные ограничения

- **Допустимые кривые**: только `‖t‖ = const`. Эллипс, парабола и другие
  кривые с переменной нормой касательной вызовут расходимость контроллера.
  См. TODO в `CLAUDE.md` (баг s_arc).

- **Размер датасета**: при `--num-curves 200 --samples 5` реально сохраняется
  меньше 1000 точек — часть кривых не проходит валидацию, часть точек
  нестабильна. Для продакшн-обучения рекомендуется `--num-curves 1000+`.

- **Скорость генерации**: каждая точка запускает несколько полных симуляций
  (oracle). Для 200 кривых × 5 точек × ~10 ролаутов ≈ 10 000 симуляций.
  На CPU занимает несколько минут.
