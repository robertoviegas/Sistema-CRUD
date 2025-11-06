from __future__ import annotations

from typing import Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_data(
    train_data: pd.DataFrame, params: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera X e Y a partir do dataset de treino.

    Args:
        train_data: DataFrame com os dados de treino
        params: Parâmetros de configuração

    Returns:
        Tupla (X, y) onde X são as features e y é o target (SalePrice)
    """
    # Identificar coluna target
    target_col = params.get("target_column", "SalePrice")

    # Separar features e target
    y = train_data[target_col].values.astype(np.float64)

    # Remover colunas não numéricas e o target
    X = train_data.drop(columns=[target_col, "Id"], errors="ignore")

    # Selecionar apenas colunas numéricas
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    # Preencher valores NaN com 0
    X = X.fillna(0)

    # Converter para numpy array
    X = X.values.astype(np.float64)

    return X, y


def split_data(
    X: np.ndarray, y: np.ndarray, params: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide os dados em conjuntos de treino e teste.

    Args:
        X: Array com as features
        y: Array com o target
        params: Parâmetros de configuração

    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    test_size = params.get("test_size", 0.2)
    random_state = params.get("seed", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def train_model(X: np.ndarray, y: np.ndarray, params: Dict):
    flavor = params.get("flavor", "sklearn")
    if flavor == "sklearn":
        from sklearn.linear_model import LinearRegression

        model = LinearRegression().fit(X, y)
        return model
    else:
        raise ValueError("Unsupported flavor. Only 'sklearn' is supported.")


def evaluate_model(
    model, X: np.ndarray, y: np.ndarray, params: Dict
) -> Dict[str, float]:
    flavor = params.get("flavor", "sklearn")
    if flavor == "sklearn":
        from sklearn.metrics import mean_squared_error, r2_score

        pred = model.predict(X)
        mse = float(mean_squared_error(y, pred))
        r2 = float(r2_score(y, pred))

        # Calcular MAPE (Mean Absolute Percentage Error)
        # Evita divisão por zero usando máscara
        mask = y != 0
        if np.any(mask):
            mape = float(np.mean(np.abs((y[mask] - pred[mask]) / y[mask])) * 100)
        else:
            # Se todos os valores são zero, usar um valor muito grande ao invés de inf
            mape = 1e10

        # Calcular MEAPE (Mean Error Absolute Percentage Error)
        # Similar ao MAPE, mas usando erro absoluto médio dividido pela média dos valores verdadeiros
        mean_abs_y = np.mean(np.abs(y))
        if mean_abs_y != 0:
            meape = float(np.mean(np.abs(y - pred)) / mean_abs_y * 100)
        else:
            # Se a média dos valores absolutos é zero, usar um valor muito grande
            meape = 1e10

        return {
            "mse": mse,
            "r2": r2,
            "mape": mape,
            "meape": meape,
        }
    else:
        raise ValueError("Unsupported flavor. Only 'sklearn' is supported.")


def log_to_mlflow(
    model, metrics: Dict[str, float], params: Dict
) -> Dict[str, str | float]:
    tracking_uri = params.get("mlflow_tracking_uri")

    # Verificar se o servidor MLflow está acessível antes de tentar usar
    if tracking_uri:
        try:
            # Testar conexão rápida com timeout curto
            import urllib.error
            import urllib.request
            from urllib.parse import urlparse

            parsed = urlparse(tracking_uri)
            # Tentar acessar a raiz do MLflow (não precisa ser /health, pode ser qualquer endpoint)
            test_url = f"{parsed.scheme}://{parsed.netloc}/"
            req = urllib.request.Request(test_url)
            req.add_header("Connection", "close")
            urllib.request.urlopen(req, timeout=2)
            # Se chegou aqui, servidor está acessível
            mlflow.set_tracking_uri(tracking_uri)
        except Exception:
            # Servidor não acessível, usar backend local imediatamente
            mlflow.set_tracking_uri("./mlruns")
    else:
        # Se não houver URI, usar backend local
        mlflow.set_tracking_uri("./mlruns")

    try:
        with mlflow.start_run(run_name="kedro-train") as run:
            mlflow.log_params(
                {
                    "flavor": params.get("flavor"),
                    "n_features": int(params.get("n_features", 3)),
                    "n_samples": int(params.get("n_samples", 200)),
                    "test_size": float(params.get("test_size", 0.2)),
                }
            )
            mlflow.log_metrics(metrics)
            flavor = params.get("flavor", "sklearn")
            if flavor == "sklearn":
                import mlflow.sklearn as msk

                # Criar input_example usando o número real de features do modelo
                n_features = getattr(
                    model, "n_features_in_", int(params.get("n_features", 3))
                )
                input_example = [[0.0] * n_features]

                msk.log_model(
                    model,
                    artifact_path="model",
                    input_example=input_example,
                )
            else:
                raise ValueError("Unsupported flavor. Only 'sklearn' is supported.")
            run_id = run.info.run_id
            model_uri = mlflow.get_artifact_uri("model")
            version = run_id[:8]
            return {
                "mlflow_run_id": run_id,
                "version": version,
                "model_uri": model_uri,
                **metrics,
            }
    except Exception:
        # Se falhar ao conectar ao servidor MLflow, tentar usar backend local
        try:
            mlflow.set_tracking_uri("./mlruns")
            with mlflow.start_run(run_name="kedro-train") as run:
                mlflow.log_params(
                    {
                        "flavor": params.get("flavor"),
                        "n_features": int(params.get("n_features", 3)),
                        "n_samples": int(params.get("n_samples", 200)),
                        "test_size": float(params.get("test_size", 0.2)),
                    }
                )
                mlflow.log_metrics(metrics)
                flavor = params.get("flavor", "sklearn")
                if flavor == "sklearn":
                    import mlflow.sklearn as msk

                    # Criar input_example usando o número real de features do modelo
                    n_features = getattr(
                        model, "n_features_in_", int(params.get("n_features", 3))
                    )
                    input_example = [[0.0] * n_features]

                    msk.log_model(
                        model,
                        artifact_path="model",
                        input_example=input_example,
                    )
                run_id = run.info.run_id
                model_uri = mlflow.get_artifact_uri("model")
                version = run_id[:8]
                return {
                    "mlflow_run_id": run_id,
                    "version": version,
                    "model_uri": model_uri,
                    **metrics,
                }
        except Exception as e2:
            # Se falhar completamente, retornar métricas sem MLflow
            import uuid

            run_id = uuid.uuid4().hex
            version = run_id[:8]
            return {
                "mlflow_run_id": None,
                "version": version,
                "model_uri": None,
                "mlflow_error": str(e2)[:200] if e2 else "Unknown error",
                **metrics,
            }
