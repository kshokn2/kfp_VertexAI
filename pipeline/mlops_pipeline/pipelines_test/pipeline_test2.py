"""
테스트 파이프라인
"""

import os
from kfp import dsl

from components import component_test

@dsl.pipeline(name="composer-test-pipeline2")
def pipeline(
    project: str = os.environ.get("VERTEX_PROJECT_ID"),
    location: str = os.environ.get("VERTEX_LOCATION"),
) -> str:

    # 학습을 위한 데이터 다운로드
    test_op = component_test.test2()

    return test_op.output
