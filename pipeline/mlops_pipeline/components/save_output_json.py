"""
E2E 파이프라인의 최종 결과를 json으로 저장하기 위한 Custom Kubeflow Component를 정의합니다.
"""

from get_env import BASE_IMAGE_URI
from kfp.dsl import Input, Metrics, component

@component(
    base_image=BASE_IMAGE_URI,
)
def save_e2e_output(
    deploy_result: str,
    deploy_model: str,
    any_artifacts: Input[Metrics],
):
    """
    Airflow(GCP의 Composer)에서 재학습 PipelineJob 실행 후 최종 결과를 가져오기 위해서 JSON 파일을 저장하는 Operator

    Args:
        deploy_result (str): 최대 3번 학습하는 재학습 파이프라인의 최종 결과.
        deploy_model (str): deploy_result가 'approved'일 때의 최종 모델 파일 위치.
        any_artifacts (Input[Metrics]): 아무 Artifact를 가져오고 artifact.path를 통해 상대적인 Job 실행 결과가 저장되는 위치의 JSON 파일을 저장하기 위해 활용하는 Input Artifact.
    """
    import os
    import json

    print("any_artifacts.path 확인 ->", any_artifacts.path)

    save_path = f"{any_artifacts.path}/../"
    os.makedirs(save_path, exist_ok=True)
    print(f"최종 결과 저장 위치: {save_path}")

    json_file = "deployment.json"
    output_dict = {
        "deploy_result": deploy_result,
        "deploy_model": deploy_model
    }

    with open(f"{save_path}/{json_file}", "w") as f:
        json.dump(output_dict, f)

    print("✅ deployment.json 저장 완료.")
