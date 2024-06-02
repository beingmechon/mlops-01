from kedro.pipeline import Pipeline, node
from .nodes import find_best_model, split_data
from kedro.io import DataCatalog

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_data,
                inputs="features",
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node"
            ),
            node(
                func=find_best_model,
                inputs=["X_train", "y_train", "param_grids"],
                outputs="model",
                name="find_best_model_node"
            )
        ]
    )
