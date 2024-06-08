from kedro.pipeline import Pipeline, node
from .nodes import train_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_model,
            inputs=["train_dataloader", 
                    "params:hidden_size",
                    "params:drop_out",
                    "params:learning_rate",
                    "params:weight_decay",
                    "params:patience",
                    "params:epochs",
                    "params:char_to_idx",
                    "params:idx_to_char",
                    "params:clip_norm",],
            outputs="trained_model",
            name="train_model_node"
        )
    ])