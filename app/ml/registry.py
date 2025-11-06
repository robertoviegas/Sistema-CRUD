import uuid
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LinearRegression


class ModelRegistryAdapter:
    def __init__(self, flavor: str, model_uri: Optional[str] = None):
        self.flavor = flavor
        self.model_uri = model_uri

    def load_active(self):
        # Se houver um model_uri (ex.: runs:/<run_id>/model), tenta carregar do MLflow
        if self.model_uri:
            try:
                if self.flavor == "sklearn":
                    import mlflow.sklearn as msk

                    return msk.load_model(self.model_uri)
                else:
                    raise ValueError("Unsupported flavor: %s" % self.flavor)
            except Exception:
                # fallback para dummy
                pass
        # Fallback: modelo dummy
        if self.flavor == "sklearn":
            model = LinearRegression()
            model.coef_ = np.array([1.0])
            model.intercept_ = 0.0
            model.n_features_in_ = 1
            return model
        else:
            raise ValueError("Unsupported flavor: %s" % self.flavor)

    def _align_features(self, xs: np.ndarray, expected: int) -> np.ndarray:
        if xs.shape[1] > expected:
            return xs[:, :expected]
        if xs.shape[1] < expected:
            pad = np.zeros((xs.shape[0], expected - xs.shape[1]))
            return np.concatenate([xs, pad], axis=1)
        return xs

    def predict(self, model, features: Dict[str, Any]) -> float:
        # Converte valores numéricos; ignora não numéricos com fallback zero
        vals = []
        for v in features.values():
            try:
                vals.append(float(v))
            except Exception:
                vals.append(0.0)
        xs = np.array([vals], dtype=float)
        if self.flavor == "sklearn":
            expected = int(getattr(model, "n_features_in_", xs.shape[1]))
            xs = self._align_features(xs, expected)
            y = model.predict(xs)
            return float(y[0])
        else:
            raise ValueError("Invalid flavor. Only 'sklearn' is supported.")

    def new_prediction_id(self) -> str:
        return uuid.uuid4().hex
