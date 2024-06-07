"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline
# from kedro.framework.project import find_pipelines

from captcha_recognition_torch.pipelines.data_processing import create_pipeline as dp
from captcha_recognition_torch.pipelines.data_loading import create_pipeline as dl
from captcha_recognition_torch.pipelines.data_science.pipeline import create_pipeline as ds

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    dp_pipeline = dp()
    dl_pipeline = dl()
    ds_pipeline = ds()

    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    pipelines = {
        "__default__": dp_pipeline + dl_pipeline + ds_pipeline,
        "dp": dp_pipeline,
        "dl": dl_pipeline,
        "ds": ds_pipeline,
    }

    return pipelines


# def register_pipelines() -> Dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines