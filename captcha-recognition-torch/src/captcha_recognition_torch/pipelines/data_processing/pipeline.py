from kedro.pipeline import Pipeline, node
from .nodes import extract_zip, split_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=extract_zip,
            inputs=["raw_data", "intermediate_data"],
            outputs=None,
            name="extract_zip_node"
        ),
        node(
            func=split_data,
            inputs=["intermediate_data", "primary_data", "params:split_ratio", "params:randome_seed"],
            outputs=None,
            name="split_data_node"
        )
    ])