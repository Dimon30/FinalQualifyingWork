"""Модели предсказания оптимальной скорости V*.

Кодовые имена:
    'mlp'  — SpeedMLP        (supervised MSE)
    'sac'  — SpeedSAC        (Gaussian NLL + entropy + twin critics)
    'td3'  — SpeedTD3        (BC + Q-guided actor + Polyak targets)
    'ppo'  — SpeedPPO        (clipped surrogate + value + entropy)

Быстрый старт::

    from ml.models.registry import get_speed_model, SpeedPredictorAny
    model = get_speed_model("sac")                          # создать новый экземпляр
    pred  = SpeedPredictorAny.load("path/to/sac_model.pt") # загрузить обученный
    v     = pred.predict(feature_vector(...))               # предсказать V*
"""
from ml.models.registry import (
    get_speed_model,
    save_speed_model_any,
    load_speed_model_any,
    SpeedPredictorAny,
    MODEL_NAMES,
)

__all__ = [
    "get_speed_model",
    "save_speed_model_any",
    "load_speed_model_any",
    "SpeedPredictorAny",
    "MODEL_NAMES",
]
