from kedro.pipeline import Pipeline, node

from .nodes import get_dataloader

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=get_dataloader,
                inputs=["train_folder", "test_folder","params:batch_size", "params:shuffle"],
                outputs=["train_dataloader","test_dataloader"],
                name="data_loader_node"
            )
        ]
    )