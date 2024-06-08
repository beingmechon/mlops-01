"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline
# from kedro.framework.project import find_pipelines

from captcha_recognition_torch.pipelines.data_processing import create_pipeline as dp
from captcha_recognition_torch.pipelines.data_loading import create_pipeline as dl
from captcha_recognition_torch.pipelines.model_training import create_pipeline as mtrain
from captcha_recognition_torch.pipelines.model_testing import create_pipeline as mtest

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    dp_pipeline = dp()
    dl_pipeline = dl()
    mtrain_pipeline = mtrain()
    mtest_pipeline = mtest()

    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    pipelines = {
        "__default__": dp_pipeline + dl_pipeline + mtrain_pipeline + mtest_pipeline,
        "dp": dp_pipeline,
        "dl": dl_pipeline,
        "train": dl_pipeline + mtrain_pipeline,
        "test": dl_pipeline + mtest_pipeline,
        "train_test": dl_pipeline + mtrain_pipeline + mtest_pipeline,

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