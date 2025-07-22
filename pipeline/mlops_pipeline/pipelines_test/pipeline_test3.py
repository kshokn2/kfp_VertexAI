"""
테스트 파이프라인
"""

import os
from kfp import dsl

from components import component_test
from typing import NamedTuple

@dsl.pipeline(name="composer-test-pipeline3")
def pipeline(
    project: str = os.environ.get("VERTEX_PROJECT_ID"),
    location: str = os.environ.get("VERTEX_LOCATION"),
) -> NamedTuple('pipeline_outputs', result=str, model_path=str):

    import random
    eval_result = "approved" if random.random() <= 0.6 else "rejected"

    output = NamedTuple('pipeline_outputs', result=str, model_path=str)
    output = output(result=eval_result, model_path="")

    # 학습을 위한 데이터 다운로드
    test_op2 = component_test.test2()
    test_op3 = component_test.test3()

    return (output.result, test_op2.output, test_op3.output)
