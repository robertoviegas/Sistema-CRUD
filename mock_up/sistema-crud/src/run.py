from __future__ import annotations

from pathlib import Path
from typing import Dict

from kedro.io import DataCatalog, MemoryDataSet
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import CSVDataSet
from sistema_crud.pipelines.train.pipeline import create_pipeline


def run_training_kedro(
    flavor: str, tracking_uri: str | None = None
) -> Dict[str, str | float]:
    pipeline = create_pipeline()

    # Determinar o caminho do arquivo CSV relativo ao diretório do projeto
    # run.py está em sistema-crud/src/, então precisamos subir dois níveis para chegar em sistema-crud/
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "01_raw" / "train.csv"

    catalog = DataCatalog(
        data_sets={
            "train_data": CSVDataSet(filepath=str(csv_path)),
            "X": MemoryDataSet(),
            "y": MemoryDataSet(),
            "model": MemoryDataSet(),
            "metrics": MemoryDataSet(),
            "train_result": MemoryDataSet(),
            "params:train": {
                "flavor": flavor,
                "mlflow_tracking_uri": tracking_uri,
                "target_column": "SalePrice",
                "test_size": 0.2,
                "seed": 42,
            },
        }
    )
    runner = SequentialRunner()
    runner.run(pipeline, catalog)
    result = catalog.get("train_result")
    return {
        "mlflow_run_id": result["mlflow_run_id"],
        "version": result["version"],
        "model_path": result.get("model_uri"),
        "mse": float(result.get("mse", 0.0)),
        "r2": float(result.get("r2", 0.0)),
        "mape": float(result.get("mape", 0.0)),
        "meape": float(result.get("meape", 0.0)),
    }
