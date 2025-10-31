from __future__ import annotations

from typing import Dict, Tuple

import mlflow
import numpy as np


def generate_data(params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(params.get("seed", 42)))
    n_samples = int(params.get("n_samples", 200))
    n_features = int(params.get("n_features", 3))
    noise = float(params.get("noise", 0.1))
    X = rng.normal(size=(n_samples, n_features))
    w = np.array(params.get("weights", [1.5, -2.0, 0.5][:n_features]))
    y = X @ w + noise * rng.normal(size=(n_samples,))
    return X, y


def train_model(X: np.ndarray, y: np.ndarray, params: Dict):
    flavor = params.get("flavor", "sklearn")
    if flavor == "sklearn":
        from sklearn.linear_model import LinearRegression

        model = LinearRegression().fit(X, y)
        return model
    elif flavor == "tensorflow":
        import tensorflow as tf

        model = tf.keras.Sequential(
            [tf.keras.layers.Input((X.shape[1],)), tf.keras.layers.Dense(1)]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=int(params.get("epochs", 5)), verbose=0)
        return model
    else:
        raise ValueError("Unsupported flavor")


def evaluate_model(
    model, X: np.ndarray, y: np.ndarray, params: Dict
) -> Dict[str, float]:
    flavor = params.get("flavor", "sklearn")
    if flavor == "sklearn":
        from sklearn.metrics import mean_squared_error

        pred = model.predict(X)
        mse = float(mean_squared_error(y, pred))
        return {"mse": mse}
    elif flavor == "tensorflow":
        mse = float(model.evaluate(X, y, verbose=0))
        return {"mse": mse}
    else:
        raise ValueError("Unsupported flavor")


def log_to_mlflow(
    model, metrics: Dict[str, float], params: Dict
) -> Dict[str, str | float]:
    tracking_uri = params.get("mlflow_tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_name="kedro-train") as run:
        mlflow.log_params(
            {
                "flavor": params.get("flavor"),
                "n_features": int(params.get("n_features", 3)),
                "n_samples": int(params.get("n_samples", 200)),
            }
        )
        mlflow.log_metrics(metrics)
        flavor = params.get("flavor", "sklearn")
        if flavor == "sklearn":
            import mlflow.sklearn as msk

            msk.log_model(model, artifact_path="model")
        elif flavor == "tensorflow":
            import mlflow.tensorflow as mtf

            mtf.log_model(model, artifact_path="model")
        else:
            raise ValueError("Unsupported flavor")
        run_id = run.info.run_id
        model_uri = mlflow.get_artifact_uri("model")
        version = run_id[:8]
        return {
            "mlflow_run_id": run_id,
            "version": version,
            "model_uri": model_uri,
            **metrics,
        }
