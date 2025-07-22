"""
테스트 파이프라인
"""

import os
from kfp import dsl

from components import component_test

@dsl.pipeline(name="composer-test-pipeline1")
def test_pipeline(
    project: str = os.environ.get("VERTEX_PROJECT_ID"),
    location: str = os.environ.get("VERTEX_LOCATION"),
):
    a = 1