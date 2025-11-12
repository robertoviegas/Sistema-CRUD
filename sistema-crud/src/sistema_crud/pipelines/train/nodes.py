from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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
        model = LinearRegression().fit(X, y)
        return model
    else:
        raise ValueError("Unsupported flavor. Only 'sklearn' is supported.")


def evaluate_model(
    model, X: np.ndarray, y: np.ndarray, params: Dict
) -> Dict[str, float]:
    flavor = params.get("flavor", "sklearn")
    if flavor == "sklearn":
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


def save_model_local(
    model, metrics: Dict[str, float], params: Dict
) -> Dict[str, str | float]:
    """
    Salva o modelo localmente usando joblib e retorna as métricas e informações do modelo.
    
    Args:
        model: Modelo treinado
        metrics: Dicionário com as métricas calculadas
        params: Parâmetros de configuração
        
    Returns:
        Dicionário com version, model_path e métricas
    """
    flavor = params.get("flavor", "sklearn")
    
    # Gerar ID único para esta execução
    run_id = uuid.uuid4().hex
    version = run_id[:8]
    
    # Definir caminho para salvar o modelo
    # Usar diretório models/ na raiz do projeto sistema-crud
    project_root = Path(__file__).parent.parent.parent.parent
    models_dir = project_root / "data" / "06_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Nome do arquivo do modelo baseado no version
    model_filename = f"model_{version}.pkl"
    model_path = models_dir / model_filename
    
    try:
        if flavor == "sklearn":
            # Salvar modelo usando joblib
            joblib.dump(model, model_path)
        else:
            raise ValueError("Unsupported flavor. Only 'sklearn' is supported.")
        
        # Retornar informações do modelo e métricas
        return {
            "version": version,
            "model_path": str(model_path),
            **metrics,
        }
    except Exception as e:
        # Se falhar ao salvar, ainda retornar métricas
        return {
            "version": version,
            "model_path": None,
            "save_error": str(e)[:200] if e else "Unknown error",
            **metrics,
        }
