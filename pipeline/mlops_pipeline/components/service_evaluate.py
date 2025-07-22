"""
kfp.dsl.component를 이용해서 서비스 평가에 관련된 Custom Kubeflow Component를 정의합니다.
(CI/CD 자동화를 위해) 평가가 관련된 코드는 Training Container URI를 통해서 관리됩니다. 따라서 평가 관련 소스 코드는 Kubeflow에 종속되지 않습니다.
평가는 evaluate_metric 컴포넌트에서 aiplatform.CustomJob을 통해 평가 Custom Job을 실행(run)하고, 평가가 종료되면 Metrics Artifact의 Metadata를 업데이트합니다.
"""

from get_env import BASE_IMAGE_URI
from kfp import dsl
from kfp.dsl import Dataset, Input, InputPath, Metrics, Model, Output, component


@component(
    base_image=BASE_IMAGE_URI,
)
def evaluate_metric(
    project: str,
    location: str,
    job_name_suffix: str,
    input_data: Input[Dataset],
    input_db: InputPath("Any"),
    input_model: Input[Model],
    input_groundtruth: InputPath("Any"),
    eval_container_uri: str,
    staging_bucket: str,
    deployment_status: Output[Metrics],
    update_case: str = "biweekly",
    job_name_prefix: str = "Evaluation CustomJob",
    replica_count: int = 1,
    eval_machine: str = "A100",
) -> str:
    """
    정답지(GroundTruth)를 기반으로 유사도 검사를 진행 후, 서비스 성능을 평가하는 Operator

    Args:
        project(str): project id of the Google Cloud project.
        location(str): location of the Google Cloud project.
        job_name_suffix(str): Job 이름 및 폴더명, 그리고 description 등에 시간 정보제공을 위한 suffix.
        input_data(Input[Dataset]): 평가에 사용하려는 Dataset Artifact.
        input_db(InputPath("Any")): 평가에 사용되는 DB Path에 대한 파라미터.
        input_model(Input[Model]): 평가에 사용되는 모델 Artifact.
        input_groundtruth(InputPath("Any")): 평가에 사용되는 정답 파일 Path에 대한 파라미터.
        eval_container_uri(str): 평가에 사용될 컨테이너 이미지.
        staging_bucket(str): 평가 platform(CustomJob 실행 시) 결과가 저장되는 GCS bucket.
        update_case(str): 시간간격 오픈된 flag로, "biweekly"는 신규 데이터로 일부 평가용 쿼리 이미지 샘플링 실시.
        job_name_prefix(str): Vertex AI Training Pipeline(또는 Custom Jobs) Job의 이름.
        replica_count(int): Job 실행 worker의 복제본 수.
        eval_machine(str): 평가를 실행하려는 Accelerator 타입.

    Return:
        deployment_status(Output[Metrics]): 평가 결과 메트릭 Artifact.
        Output(str): 평가 컴포넌트의 return값으로 "approved" 또는 "rejected"값을 가지며 평가 결과에 대한 파이프라인과 동일.
    """
    import os
    import sys
    import json
    from pathlib import Path
    from google.cloud import aiplatform

    machine_spec = {
        "CPU": {
            "machine_type": "n1-standard-16",
            "accelerator_type": "ACCELERATOR_TYPE_UNSPECIFIED",
            "accelerator_count": 0,
        },
        "A100": {
            "machine_type": "a2-highgpu-1g",
            "accelerator_type": "NVIDIA_TESLA_A100",
            "accelerator_count": 1,
        },
        "L4": {
            "machine_type": "g2-standard-4",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1,
        },
        "T4": {
            "machine_type": "n1-standard-4",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
        }
    }

    if eval_machine not in machine_spec.keys():
        eval_machine = "L4"

    cmd_args = [
        "--project", project,
        "--update_case", update_case,
        "--input_path", input_data.path,
        "--input_db", input_db,
        "--input_model", input_model.path if os.path.isfile(input_model.path + ".pth") and not os.path.isfile(os.path.join(Path(input_model.path).parent, "pretrained")) else "pretrained",
        "--input_gt", input_groundtruth,
        "--output_result", deployment_status.path,
    ]

    aiplatform.init(project=project, location=location)

    worker_pool_specs = [
        {
            "machine_spec": machine_spec[eval_machine],
            "replica_count": replica_count,
            "container_spec": {
                "image_uri": eval_container_uri,
                "command": ["python3", "pipeline/service-evaluate/main.py"],
                "args": cmd_args,
            }
        }
    ]

    job = aiplatform.CustomJob(
        display_name=f"{job_name_prefix} {job_name_suffix}",
        worker_pool_specs=worker_pool_specs,
        staging_bucket=staging_bucket,
    )

    job.run()

    if len(os.listdir(deployment_status.path)) > 0:
        eval_result = os.listdir(deployment_status.path)[0]

        deployment_status.metadata["result"] = eval_result

        with open(os.path.join(deployment_status.path, os.listdir(deployment_status.path)[0]), "r") as fp:
            parsed_metrics = json.load(fp)

        for k, v in parsed_metrics.items():
            deployment_status.log_metric(k, v)

        return eval_result if eval_result == "approved" or eval_result == "rejected" else "rejected"
    else:
        print("결과파일 저장안됨. 메트릭 아티팩트 확인")
        sys.exit(1)
