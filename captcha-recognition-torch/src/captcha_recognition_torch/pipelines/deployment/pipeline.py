from kedro.pipeline import Pipeline, node
from .nodes import run_app

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_app,
                inputs=["trained_model", "params:idx_to_char"],
                outputs=None,
                name="run_app_node",
            )
        ]
    )
