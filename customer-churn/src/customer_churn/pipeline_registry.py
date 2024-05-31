# """Project pipelines."""
# from typing import Dict

# from kedro.framework.project import find_pipelines
# from kedro.pipeline import Pipeline


# def register_pipelines() -> Dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines

from typing import Dict
from kedro.pipeline import Pipeline
from customer_churn.pipelines.data_engineering import pipeline as de
from customer_churn.pipelines.feature_engineering import pipeline as fe
from customer_churn.pipelines.modeling import pipeline as model

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Register individual pipelines
    data_engineering_pipeline = de.create_pipeline()
    feature_engineering_pipeline = fe.create_pipeline()
    modeling_pipeline = model.create_pipeline()

    # Combine all pipelines
    pipelines = {
        "de": data_engineering_pipeline,
        "fe": feature_engineering_pipeline,
        "model": modeling_pipeline,
        "__default__": data_engineering_pipeline + feature_engineering_pipeline + modeling_pipeline
    }
    
    return pipelines
