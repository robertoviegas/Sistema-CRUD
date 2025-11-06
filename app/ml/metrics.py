import math
from typing import Dict, Optional


def compute_per_prediction_metrics(
    y_pred: float, features: Dict, y_true: Optional[float] = None
) -> Dict[str, float]:
    metrics = {
        "prediction_abs": abs(y_pred),
        "features_l2": math.sqrt(sum((float(v) ** 2 for v in features.values()))),
    }
    if y_true is not None:
        err = float(y_pred - float(y_true))
        metrics.update(
            {
                "error_abs": abs(err),
                "error_sq": err * err,
            }
        )
    # robustez simples: faixa e NaNs
    try:
        is_large = abs(y_pred) > 1e6
        metrics["robust_is_prediction_large"] = 1.0 if is_large else 0.0
    except Exception:
        metrics["robust_is_prediction_large"] = 1.0
    metrics["robust_has_nan_feature"] = (
        1.0 if any(_is_nan(v) for v in features.values()) else 0.0
    )
    return metrics


def _is_nan(v) -> bool:
    try:
        return float(v) != float(v)
    except Exception:
        return False
