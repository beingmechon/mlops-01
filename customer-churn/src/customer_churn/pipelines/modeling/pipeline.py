from kedro.pipeline import Pipeline, node
from .nodes import train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_model,
                inputs=["model", "X_train", "y_train", "X_test", "y_test", "params:columns", "params:cf"],
                outputs="training_results",
                name="train_model_node"
            ),
            node(
                func=evaluate_model,
                inputs="training_results",
                outputs="evaluation_metrics",
                name="evaluate_model_node"
            )
        ]
    )
