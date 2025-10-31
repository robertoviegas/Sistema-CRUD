from __future__ import annotations

from pathlib import Path
from typing import Dict

from kedro.io import DataCatalog, MemoryDataSet
from kedro.runner import SequentialRunner

from sistema_crud.pipelines.train.pipeline import create_pipeline


def run_training_kedro(
    flavor: str, tracking_uri: str | None = None
) -> Dict[str, str | float]:
    pipeline = create_pipeline()
    catalog = DataCatalog(
        {
            "X": MemoryDataSet(),
            "y": MemoryDataSet(),
            "model": MemoryDataSet(),
            "metrics": MemoryDataSet(),
            "train_result": MemoryDataSet(),
            "params:train": {
                "flavor": flavor,
                "mlflow_tracking_uri": tracking_uri,
                "n_samples": 200,
                "n_features": 3,
                "noise": 0.1,
                "epochs": 5,
                "seed": 42,
            },
        }
    )
    runner = SequentialRunner()
    runner.run(pipeline, catalog)
    result = catalog.load("train_result")
    return {
        "mlflow_run_id": result["mlflow_run_id"],
        "version": result["version"],
        "model_path": result.get("model_uri"),
        "mse": float(result.get("mse", 0.0)),
    }
