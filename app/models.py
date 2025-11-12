from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .db import Base


class ModelRegistry(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    flavor = Column(String(50), nullable=False)
    version = Column(String(100), nullable=False)
    model_path = Column(String(500), nullable=True)  # Caminho do modelo salvo localmente
    mlflow_run_id = Column(String(100), nullable=True)  # Mantido apenas para compatibilidade com dados antigos
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(String(64), primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    features = Column(JSON, nullable=False)
    prediction = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    model = relationship("ModelRegistry")
    metrics = relationship(
        "PredictionMetric",
        back_populates="prediction_obj",
        cascade="all, delete-orphan",
    )


class PredictionMetric(Base):
    __tablename__ = "prediction_metrics"
    id = Column(Integer, primary_key=True)
    prediction_id = Column(String(64), ForeignKey("predictions.id"), nullable=False)
    name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)

    prediction_obj = relationship("Prediction", back_populates="metrics")


class Retraining(Base):
    __tablename__ = "retrainings"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    triggered_by = Column(String(50), nullable=False)
    notes = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    model = relationship("ModelRegistry")

