from kedro.pipeline import Pipeline, node
from .nodes import evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([ 
        node(
            func=evaluate_model,
            inputs=["test_dataloader", 
                    "trained_model",
                    "params:idx_to_char"],
            outputs="result_json",
            name="evaluate_model_node"
        )
    ])