import os
import sys
import time

from flask import Flask, jsonify, request
from flask_swagger_ui import get_swaggerui_blueprint
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .config import Settings
from .db import get_engine, get_session_factory
from .ml.metrics import compute_per_prediction_metrics
from .ml.registry import ModelRegistryAdapter
from .models import ModelRegistry, Prediction, PredictionMetric, Retraining
from .schemas import PredictRequest, PredictResponse

try:
    from sistema_crud.src.run import run_training_kedro  # se instalado como pacote
except Exception:
    from run import run_training_kedro

# garantir que o path do Kedro (sistema-crud/src) esteja disponível
_KEDRO_SRC = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "sistema-crud", "src"
)
if _KEDRO_SRC not in sys.path:
    sys.path.append(_KEDRO_SRC)
# fallback direto do src


settings = Settings()
engine = get_engine(settings.db_url)
SessionFactory = get_session_factory(settings.db_url)


def register_routes(app: Flask) -> None:
    # Configurar Swagger UI
    SWAGGER_URL = "/swagger"
    API_URL = "/openapi.json"

    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL, API_URL, config={"app_name": "Sistema CRUD API"}
    )
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

    # Endpoint para servir a especificação OpenAPI
    @app.get("/openapi.json")
    def openapi_spec():
        from .openapi_spec import get_openapi_spec

        # Atualizar a URL do servidor dinamicamente baseado na requisição
        spec = get_openapi_spec()
        # Usar a URL da requisição para o servidor
        request_url = request.url_root.rstrip("/")
        spec["servers"] = [{"url": request_url, "description": "Servidor atual"}]
        return jsonify(spec)

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
                    flavor=settings.model_flavor, version="v0", model_path=None
                )
                session.add(model_row)
                session.commit()
                session.refresh(model_row)

            adapter = ModelRegistryAdapter(model_row.flavor, model_row.model_path)
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

            session.commit()

        resp = PredictResponse(
            prediction_id=pred_id,
            prediction=y_pred,
            model_id=model_row.id,
            metrics=[{"name": k, "value": float(v)} for k, v in metrics_map.items()],
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
                        "model_path": r.model_path,
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
        if new_flavor not in {"sklearn"}:
            return jsonify({"error": "flavor must be sklearn"}), 400
        with Session(engine) as session:
            row = ModelRegistry(flavor=new_flavor, version="v0", model_path=None)
            session.add(row)
            session.commit()
            return jsonify({"model_id": row.id, "flavor": row.flavor})

    @app.post("/train")
    def train():
        # Obter target_column do body da requisição, usar padrão se não fornecido
        body = request.get_json(force=True, silent=True) or {}
        target_column = body.get("target_column", "SalePrice")

        start_time = time.time()
        result = run_training_kedro(settings.model_flavor, target_column=target_column)
        training_time = time.time() - start_time
        with Session(engine) as session:
            # Verificar se já existe um modelo antes de criar o novo
            existing_model = (
                session.execute(select(ModelRegistry).order_by(ModelRegistry.id.desc()))
                .scalars()
                .first()
            )
            is_retraining = existing_model is not None

            model_path = result.get("model_path")
            row = ModelRegistry(
                flavor=settings.model_flavor,
                version=str(result.get("version", "unknown")),
                model_path=str(model_path) if model_path is not None else None,
            )
            session.add(row)
            session.flush()

            # Criar registro de retraining apenas se já existir um modelo anterior
            retraining_id = None
            if is_retraining:
                retr = Retraining(
                    model_id=row.id, triggered_by="api", notes="kedro-retrain"
                )
                session.add(retr)
                session.commit()
                retraining_id = retr.id
            else:
                session.commit()

            # Acessar atributos antes de sair do bloco with para evitar DetachedInstanceError
            model_id = row.id
            flavor = row.flavor
            version = row.version
            model_path_value = row.model_path
        return jsonify(
            {
                "model_id": model_id,
                "flavor": flavor,
                "version": version,
                "model_path": model_path_value,
                "mse": float(result.get("mse", 0.0)),
                "r2": float(result.get("r2", 0.0)),
                "mape": float(result.get("mape", 0.0)),
                "meape": float(result.get("meape", 0.0)),
                "retraining_id": retraining_id,
                "is_retraining": is_retraining,
                "training_time_seconds": round(training_time, 2),
            }
        )
