from __future__ import annotations

from typing import Dict, List, Optional

import mlflow
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

# Buffer simples em memória para janelas de produção
_PRODUCTION_BUFFER: List[Dict] = []


def simple_drift_signal(latest_features: Dict) -> Dict[str, float]:
    # estatística simples (mantida)
    n_numeric = sum(1 for v in latest_features.values() if _is_float(v))
    return {
        "fraction_numeric": n_numeric / max(1, len(latest_features)),
    }


def push_production_sample(features: Dict, prediction: float) -> None:
    _PRODUCTION_BUFFER.append({**features, "__prediction__": prediction})


def maybe_run_evidently_and_log(
    baseline_df: Optional[pd.DataFrame],
    feature_order: Optional[List[str]],
    window_size: int,
    min_samples: int,
    mlflow_tracking_uri: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """
    Se o buffer tiver amostras suficientes, roda Evidently DataDriftPreset
    e registra artefatos HTML/JSON no MLflow. Retorna dict com uris dos artefatos.
    """
    if len(_PRODUCTION_BUFFER) < max(min_samples, window_size):
        return None
    window = _PRODUCTION_BUFFER[-window_size:]
    prod_df = pd.DataFrame(window)
    if feature_order:
        # garantir colunas em ordem
        cols = [c for c in feature_order if c in prod_df.columns]
        # manter a coluna de prediction ao fim
        prod_df = prod_df[
            cols + (["__prediction__"] if "__prediction__" in prod_df.columns else [])
        ]

    report = Report(metrics=[DataDriftPreset()])
    if baseline_df is None:
        # sem baseline, usar primeiras amostras como baseline dinâmica
        baseline_df = prod_df.copy()

    report.run(reference_data=baseline_df, current_data=prod_df)

    html_artifact = report.as_html()
    json_artifact = report.as_dict()

    # registrar no MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    with mlflow.start_run(run_name="evidently-drift", nested=True) as run:
        html_path = "evidently_drift_report.html"
        json_path = "evidently_drift_report.json"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_artifact)
        pd.Series(json_artifact).to_json(json_path)
        mlflow.log_artifact(html_path, artifact_path="monitoring")
        mlflow.log_artifact(json_path, artifact_path="monitoring")
        return {
            "mlflow_run_id": run.info.run_id,
            "html_artifact_uri": mlflow.get_artifact_uri("monitoring/" + html_path),
            "json_artifact_uri": mlflow.get_artifact_uri("monitoring/" + json_path),
        }


def _is_float(v) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False
