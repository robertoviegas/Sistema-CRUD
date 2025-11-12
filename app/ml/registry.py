import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression


class ModelRegistryAdapter:
    def __init__(self, flavor: str, model_path: Optional[str] = None):
        self.flavor = flavor
        self.model_path = model_path

    def load_active(self):
        # Se houver um model_path, tenta carregar do arquivo local
        if self.model_path and os.path.exists(self.model_path):
            try:
                if self.flavor == "sklearn":
                    return joblib.load(self.model_path)
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

    @staticmethod
    def save_model(model, model_path: str, flavor: str = "sklearn"):
        """Salva o modelo em um arquivo local usando joblib."""
        if flavor == "sklearn":
            # Criar diretório se não existir
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            return model_path
        else:
            raise ValueError("Unsupported flavor: %s" % flavor)
