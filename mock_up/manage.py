import os
import sys

import click
import pandas as pd
import requests

# Adicionar o path do sistema-crud ao sys.path
sistema_crud_path = os.path.join(os.path.dirname(__file__), "sistema-crud", "src")
if sistema_crud_path not in sys.path:
    sys.path.append(sistema_crud_path)

from run import run_training_kedro

from app import create_app
from app import routes as _routes  # ensure routes are registered
from app.config import Settings
from app.db import Base, get_engine


@click.group()
def cli():
    pass


@cli.command("run")
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000, type=int)
def run(host: str, port: int):
    app = create_app()
    app.run(host=host, port=port, debug=True)


@cli.command("init-db")
def init_db():
    settings = Settings()
    engine = get_engine(settings.db_url)
    Base.metadata.create_all(bind=engine)
    click.echo("Database initialized.")


@cli.command("train-kedro")
def train_kedro():
    settings = Settings()
    result = run_training_kedro(settings.model_flavor, settings.mlflow_tracking_uri)
    click.echo(result)


@cli.command("predict-csv")
@click.argument("csv_path")
@click.option("--url", default="http://localhost:8000/predict")
@click.option("--feature-cols", default="")
@click.option("--y-col", default="")
@click.option("--limit", default=10, type=int)
def predict_csv(csv_path: str, url: str, feature_cols: str, y_col: str, limit: int):
    """Lê um CSV e envia predições para a API."""
    df = pd.read_csv(csv_path)
    cols = [c.strip() for c in feature_cols.split(",") if c.strip()] or [
        c for c in df.columns if c != y_col
    ]
    for _, row in df.head(limit).iterrows():
        features = {c: row[c] for c in cols if c in df.columns}
        payload = {"features": features}
        if y_col and y_col in df.columns:
            payload["y_true"] = row[y_col]
        resp = requests.post(url, json=payload)
        click.echo(resp.json())


if __name__ == "__main__":
    cli()
    cli()
