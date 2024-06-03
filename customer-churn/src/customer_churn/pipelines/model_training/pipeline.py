from kedro.pipeline import Pipeline, node
from .nodes import train_model, split_data #, evaluate_model

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
                func=train_model,
                inputs=["X_train", "y_train", "X_test", "y_test", "param_grids"], # "params:cf"
                # outputs="training_results",
                outputs=["model", "training_results", "training_image"],
                name="train_model_node"
            ),
            # node(
            #     func=evaluate_model,
            #     inputs="training_results",
            #     outputs="evaluation_metrics",
            #     name="evaluate_model_node"
            # )
        ]
    )
