import os
import sys

import pandas as pd
from flask import Flask, jsonify, request
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

# garantir que o path do Kedro (sistema-crud/src) esteja disponível
_KEDRO_SRC = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "sistema-crud", "src"
)
if _KEDRO_SRC not in sys.path:
    sys.path.append(_KEDRO_SRC)
try:
    from sistema_crud.src.run import run_training_kedro  # se instalado como pacote
except Exception:
    from run import run_training_kedro  # fallback direto do src

from .config import Settings
from .db import get_engine, get_session_factory
from .ml.drift import (
    maybe_run_evidently_and_log,
    push_production_sample,
    simple_drift_signal,
)
from .ml.metrics import compute_per_prediction_metrics
from .ml.registry import ModelRegistryAdapter
from .models import ModelRegistry, Prediction, PredictionMetric, Retraining
from .schemas import PredictRequest, PredictResponse

settings = Settings()
engine = get_engine(settings.db_url)
SessionFactory = get_session_factory(settings.db_url)


# carregar baseline opcional
_baseline_df = None
_feature_order = None
if settings.evidently_baseline_csv:
    try:
        _baseline_df = pd.read_csv(settings.evidently_baseline_csv)
        if settings.evidently_feature_keys:
            _feature_order = [
                k.strip()
                for k in settings.evidently_feature_keys.split(",")
                if k.strip()
            ]
    except Exception:
        _baseline_df = None
        _feature_order = None


def register_routes(app: Flask) -> None:
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "env": settings.app_env,
        }

    @app.post("/predict")
    def predict():
        payload = request.get_json(force=True, silent=False) or {}
        y_true = payload.get("y_true")
        data = PredictRequest(**{k: v for k, v in payload.items() if k == "features"})

        with Session(engine) as session:
            model_row = (
                session.execute(select(ModelRegistry).order_by(ModelRegistry.id.desc()))
                .scalars()
                .first()
            )
            if not model_row:
                model_row = ModelRegistry(
                    flavor=settings.model_flavor, version="v0", mlflow_run_id=None
                )
                session.add(model_row)
                session.commit()
                session.refresh(model_row)

            model_uri = (
                f"runs:/{model_row.mlflow_run_id}/model"
                if model_row.mlflow_run_id
                else None
            )
            adapter = ModelRegistryAdapter(model_row.flavor, model_uri)
            model = adapter.load_active()
            y_pred = adapter.predict(model, data.features)

            pred_id = adapter.new_prediction_id()
            pred_row = Prediction(
                id=pred_id,
                model_id=model_row.id,
                features=data.features,
                prediction=y_pred,
            )
            session.add(pred_row)
            session.flush()

            metrics_map = compute_per_prediction_metrics(
                y_pred, data.features, y_true=y_true
            )
            for name, value in metrics_map.items():
                session.add(
                    PredictionMetric(
                        prediction_id=pred_row.id, name=name, value=float(value)
                    )
                )

            drift_map = simple_drift_signal(data.features)
            for name, value in drift_map.items():
                session.add(
                    PredictionMetric(
                        prediction_id=pred_row.id,
                        name=f"drift::{name}",
                        value=float(value),
                    )
                )

            session.commit()

        # Alimenta buffer de produção e dispara Evidently se possível
        push_production_sample(data.features, y_pred)
        maybe_run_evidently_and_log(
            baseline_df=_baseline_df,
            feature_order=_feature_order,
            window_size=settings.evidently_window_size,
            min_samples=settings.evidently_min_samples,
            mlflow_tracking_uri=settings.mlflow_tracking_uri,
        )

        resp = PredictResponse(
            prediction_id=pred_id,
            prediction=y_pred,
            model_id=model_row.id,
            metrics=[
                {"name": k, "value": float(v)}
                for k, v in {**metrics_map, **drift_map}.items()
            ],
        )
        return jsonify(resp.model_dump())

    @app.get("/predictions")
    def list_predictions():
        page = int(request.args.get("page", 1))
        size = min(int(request.args.get("size", 50)), 200)
        model_id = request.args.get("model_id")
        stmt = select(Prediction).order_by(Prediction.created_at.desc())
        if model_id:
            try:
                stmt = stmt.where(Prediction.model_id == int(model_id))
            except Exception:
                pass
        offset = (page - 1) * size
        with Session(engine) as session:
            rows = session.execute(stmt.offset(offset).limit(size)).scalars().all()
            return jsonify(
                [
                    {
                        "id": r.id,
                        "model_id": r.model_id,
                        "prediction": r.prediction,
                        "features": r.features,
                        "created_at": r.created_at.isoformat(),
                    }
                    for r in rows
                ]
            )

    @app.get("/metrics")
    def list_metrics():
        pred_id = request.args.get("prediction_id")
        name = request.args.get("name")
        page = int(request.args.get("page", 1))
        size = min(int(request.args.get("size", 100)), 500)
        stmt = select(PredictionMetric)
        if pred_id:
            stmt = stmt.where(PredictionMetric.prediction_id == pred_id)
        if name:
            stmt = stmt.where(PredictionMetric.name == name)
        offset = (page - 1) * size
        with Session(engine) as session:
            rows = session.execute(stmt.offset(offset).limit(size)).scalars().all()
            return jsonify(
                [
                    {
                        "id": r.id,
                        "prediction_id": r.prediction_id,
                        "name": r.name,
                        "value": r.value,
                    }
                    for r in rows
                ]
            )

    @app.get("/models")
    def list_models():
        page = int(request.args.get("page", 1))
        size = min(int(request.args.get("size", 50)), 200)
        flavor = request.args.get("flavor")
        stmt = select(ModelRegistry).order_by(ModelRegistry.created_at.desc())
        if flavor:
            stmt = stmt.where(ModelRegistry.flavor == flavor)
        offset = (page - 1) * size
        with Session(engine) as session:
            rows = session.execute(stmt.offset(offset).limit(size)).scalars().all()
            return jsonify(
                [
                    {
                        "id": r.id,
                        "flavor": r.flavor,
                        "version": r.version,
                        "mlflow_run_id": r.mlflow_run_id,
                        "created_at": r.created_at.isoformat(),
                    }
                    for r in rows
                ]
            )

    @app.get("/retrainings")
    def list_retrainings():
        page = int(request.args.get("page", 1))
        size = min(int(request.args.get("size", 50)), 200)
        stmt = select(Retraining).order_by(Retraining.created_at.desc())
        offset = (page - 1) * size
        with Session(engine) as session:
            rows = session.execute(stmt.offset(offset).limit(size)).scalars().all()
            return jsonify(
                [
                    {
                        "id": r.id,
                        "model_id": r.model_id,
                        "triggered_by": r.triggered_by,
                        "notes": r.notes,
                        "created_at": r.created_at.isoformat(),
                    }
                    for r in rows
                ]
            )

    @app.delete("/records/<string:table>/<string:item_id>")
    def delete_record(table: str, item_id: str):
        table_map = {
            "predictions": Prediction,
            "prediction_metrics": PredictionMetric,
            "models": ModelRegistry,
            "retrainings": Retraining,
        }
        if table not in table_map:
            return jsonify({"error": "invalid table"}), 400
        Model = table_map[table]

        with Session(engine) as session:
            if (
                Model is PredictionMetric
                or Model is Retraining
                or Model is ModelRegistry
            ):
                try:
                    iid = int(item_id)
                except Exception:
                    return jsonify({"error": "id must be integer"}), 400
                stmt = delete(Model).where(Model.id == iid)
            else:
                stmt = delete(Model).where(Model.id == item_id)
            res = session.execute(stmt)
            session.commit()
            return jsonify({"deleted": res.rowcount})

    @app.post("/switch-model")
    def switch_model():
        body = request.get_json(force=True) or {}
        new_flavor = body.get("flavor")
        if new_flavor not in {"sklearn", "tensorflow"}:
            return jsonify({"error": "flavor must be sklearn or tensorflow"}), 400
        with Session(engine) as session:
            row = ModelRegistry(flavor=new_flavor, version="v0", mlflow_run_id=None)
            session.add(row)
            session.commit()
            return jsonify({"model_id": row.id, "flavor": row.flavor})

    @app.post("/train")
    def train():
        result = run_training_kedro(settings.model_flavor, settings.mlflow_tracking_uri)
        with Session(engine) as session:
            row = ModelRegistry(
                flavor=settings.model_flavor,
                version=str(result.get("version")),
                mlflow_run_id=str(result.get("mlflow_run_id")),
            )
            session.add(row)
            session.flush()
            retr = Retraining(model_id=row.id, triggered_by="api", notes="kedro-train")
            session.add(retr)
            session.commit()
        return jsonify(
            {
                "model_id": row.id,
                "flavor": row.flavor,
                "version": row.version,
                "mlflow_run_id": row.mlflow_run_id,
                "mse": float(result.get("mse", 0.0)),
                "retraining_id": retr.id,
            }
        )
