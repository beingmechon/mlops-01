from kedro.pipeline import Pipeline, node
from .nodes import find_best_model
from kedro.io import DataCatalog

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=find_best_model,
                inputs=["X_train", "y_train", "param_grids"],
                outputs="model",
                name="find_best_model_node"
            )
        ]
    )
