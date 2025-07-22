"""
테스트 컴포넌트
"""

from kfp.dsl import component, Metrics, Dataset, Output, OutputPath
from get_env import BASE_IMAGE_URI


# 컴포넌트 1: CSV 생성 및 메트릭 기록
@component(
    # base_image="python:3.10-bullseye",
    base_image=BASE_IMAGE_URI,
    # base_image="asia-northeast3-docker.pkg.dev/mlops-2d00-pjt/ar-docker-repo-2d00-an3/kfp-base-image",
)
def test(
    data: Output[Dataset],
    metrics: Output[Metrics],
):
    from pathlib import Path
    import os

    metrics.log_metric("avgLoss", 0.1)
    Path(data.path).parent.mkdir(parents=True, exist_ok=True)

    csv_content = """name,age,city
Alice,30,New York
Bob,25,Los Angeles
Charlie,35,Chicago"""

    with open(data.path, "w", encoding="utf-8") as file:
        file.write(csv_content)


# 컴포넌트 2: 무작위 승인 여부 판단
@component(
    # base_image="python:3.10-bullseye",
    base_image=BASE_IMAGE_URI,
    # base_image="asia-northeast3-docker.pkg.dev/mlops-2d00-pjt/ar-docker-repo-2d00-an3/kfp-base-image",
)
def test2() -> str:
    from pathlib import Path
    import os
    import random

    eval_result = "approved" if random.random() <= 0.6 else "rejected"
    print(eval_result)

    return eval_result


# 컴포넌트 3: 모델 경로 반환
@component(
    # base_image="python:3.10-bullseye",
    base_image=BASE_IMAGE_URI,
    # base_image="asia-northeast3-docker.pkg.dev/mlops-2d00-pjt/ar-docker-repo-2d00-an3/kfp-base-image",
)
def test3() -> str:
    from pathlib import Path
    import os

    model = "gs://cs-2d00-an3a-backup-bucket/test/model.pth"
    print(model)

    return model
