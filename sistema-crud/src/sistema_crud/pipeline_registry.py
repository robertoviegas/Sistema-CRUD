"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline
from kedro.pipeline.modular_pipeline import pipeline

from sistema_crud.pipelines import train 

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    train_pipeline = train.create_pipeline()

    return {
        "__default__": train_pipeline, 
    }
 