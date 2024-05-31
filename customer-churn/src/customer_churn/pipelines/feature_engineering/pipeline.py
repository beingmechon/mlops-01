from kedro.pipeline import Pipeline, node
from .nodes import create_features

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=create_features,
            inputs="preprocessed_data",
            outputs="features",
            name="create_features_node"
        )
    ])
