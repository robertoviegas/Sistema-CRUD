from typing import Any, Dict, List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Mapa de features")


class PredictResponse(BaseModel):
    prediction_id: str
    prediction: float
    model_id: int
    metrics: List[dict]

