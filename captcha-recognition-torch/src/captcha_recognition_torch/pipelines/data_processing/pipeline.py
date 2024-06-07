from kedro.pipeline import Pipeline, node
from .nodes import extract_zip, split_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=extract_zip,
            inputs=["raw_data", "params:extract_to"],
            outputs="extract_to",
            name="extract_zip_node"
        ),
        node(
            func=split_data,
            inputs=["extract_to", "primary_data", "params:split_ratio", "params:random_seed"],
            outputs=["train_folder", "test_folder"],
            name="split_data_node"
        )
    ])