from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import evaluate_model, generate_data, log_to_mlflow, train_model


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=generate_data,
                inputs="params:train",
                outputs=["X", "y"],
                name="generate_data",
            ),
            node(
                func=train_model,
                inputs=["X", "y", "params:train"],
                outputs="model",
                name="train_model",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X", "y", "params:train"],
                outputs="metrics",
                name="evaluate_model",
            ),
            node(
                func=log_to_mlflow,
                inputs=["model", "metrics", "params:train"],
                outputs="train_result",
                name="log_to_mlflow",
            ),
        ]
    )
