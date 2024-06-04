from kedro.pipeline import Pipeline, node
from .nodes import evaluate_model
from kedro.io import DataCatalog

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test"],
                outputs=["testing_results", "testing_image"],
                name="evaluate_model_node"
            )
        ]
    )
