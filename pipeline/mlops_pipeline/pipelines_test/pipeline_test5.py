"""
테스트 파이프라인
"""

import os
from kfp import dsl
from kfp.dsl import ExitHandler, pipeline, component

from get_env import BASE_IMAGE_URI

@component(base_image=BASE_IMAGE_URI)
def log_op():
    import logging

    print("This is a print function")
    logging.info("This is a logging function in INFO")

@pipeline(name='logging-test-pipeline')
def test_pipeline(
    project: str = os.environ.get("VERTEX_PROJECT_ID"),
    location: str = os.environ.get("VERTEX_LOCATION"),
):
    task = log_op()
