from kedro.pipeline import Pipeline, node
from .nodes import extract_zip, split_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            inputs=["raw_data", "dest_dir"],
            outputs=None,
            name="extract_zip_node"
        ),
        node(
            func=split_data,
            inputs=["dest_dir", "primary_data", "params:split_ratio", "params:random_seed"],
            outputs=None,
            name="split_data_node"
        )
    ])