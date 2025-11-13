from __future__ import annotations

from pathlib import Path
from typing import Dict

from kedro.config import ConfigLoader
from kedro.io import DataCatalog, MemoryDataSet
from kedro.runner import SequentialRunner
from sistema_crud.pipelines.train.pipeline import create_pipeline


def run_training_kedro(
    flavor: str, target_column: str = "SalePrice"
) -> Dict[str, str | float]:
    pipeline = create_pipeline()

    # Determinar o caminho do projeto (sistema-crud/)
    # run.py está em sistema-crud/src/, então precisamos subir um nível
    project_root = Path(__file__).parent.parent

    # Carregar catálogo do arquivo de configuração
    # O padrão do Kedro é usar "conf" como diretório de configuração
    conf_path = project_root / "conf"
    config_loader = ConfigLoader(conf_source=str(conf_path))
    catalog_config = config_loader.get("catalog*", "catalog*/**")

    # Resolver caminhos relativos no catálogo em relação ao diretório do projeto
    # O Kedro espera que os caminhos sejam relativos ao diretório raiz do projeto
    if "train_data" in catalog_config and "filepath" in catalog_config["train_data"]:
        filepath = catalog_config["train_data"]["filepath"]
        # Se o caminho for relativo, resolver em relação ao project_root
        if not Path(filepath).is_absolute():
            catalog_config["train_data"]["filepath"] = str(project_root / filepath)

    # Criar catálogo a partir da configuração
    catalog = DataCatalog.from_config(catalog_config)

    # Adicionar datasets em memória necessários para o pipeline
    catalog.add("X", MemoryDataSet())
    catalog.add("X_train", MemoryDataSet())
    catalog.add("X_test", MemoryDataSet())
    catalog.add("y", MemoryDataSet())
    catalog.add("y_train", MemoryDataSet())
    catalog.add("y_test", MemoryDataSet())
    catalog.add("model", MemoryDataSet())
    catalog.add("metrics", MemoryDataSet())
    catalog.add("train_result", MemoryDataSet())

    # Carregar parâmetros do arquivo de configuração e atualizar com valores dinâmicos
    params_config = config_loader.get("parameters*", "parameters*/**")
    train_params = params_config.get("train", {})
    # Atualizar com valores dinâmicos
    train_params["flavor"] = flavor
    train_params["target_column"] = target_column

    # Adicionar parâmetros usando MemoryDataSet
    catalog.add("params:train", MemoryDataSet(train_params))

    runner = SequentialRunner()
    runner.run(pipeline, catalog)
    # Usar load() ao invés de get() para obter dados do catálogo
    result = catalog.load("train_result")
    return {
        "version": result.get("version", "unknown"),
        "model_path": result.get("model_path"),
        "mse": float(result.get("mse", 0.0)),
        "r2": float(result.get("r2", 0.0)),
        "mape": float(result.get("mape", 0.0)),
        "meape": float(result.get("meape", 0.0)),
    }
