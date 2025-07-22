"""
kfp.dsl.component를 이용해서 모델 학습에 관련된 Custom Kubeflow Component를 정의합니다.
(CI/CD 자동화를 위해) 학습 관련된 코드는 Training Container URI를 통해서 관리됩니다. 따라서 모델 소스 코드는 Kubeflow에 종속되지 않습니다.
학습은 custom_train_job 컴포넌트에서 aiplatform.CustomContainerTrainingJob를 통해 모델 학습 Custom Job을 실행(run)하고, 학습이 종료되면 Model Artifact의 Metadata를 업데이트합니다.
"""

from get_env import BASE_IMAGE_URI
from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output, component

@component(
    base_image=BASE_IMAGE_URI,
)
def custom_train_job(
    project: str,
    location: str,
    job_name_suffix: str,
    service_account: str,
    network: str,
    input_data: Input[Dataset],
    hparams: dict,
    tensorboard_root: str,
    train_container_uri: str,
    staging_bucket: str,
    train_data: Output[Dataset],
    valid_data: Output[Dataset],
    test_data: Output[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    tensorboard_resource_name: str = None,
    input_test_path: str = None,
    job_name_prefix: str = "CustomTrainingJob",
    replica_count: int = 1,
    train_machine: str = "T4",
) -> str:
    """
    Container 기반(aiplatform.CustomContainerTrainingJob)으로 학습을 수행하는 Operator.
    job이 run된 이후에 컨테이너 안에서 생성된 Output Artifact들의 metadata를 추가로 기록하는 구조로 개발.
    참고 URL:
    https://github.com/GoogleCloudPlatform/vertex-pipelines-end-to-end-samples/blob/main/components/vertex-components/src/vertex_components/custom_train_job.py

    Args:
        project (str): GCP project ID
        location (str): 리전 (예: asia-northeast3)
        job_name_suffix (str): Job 이름 및 timestamp 등 구분자
        service_account (str): 실행할 service account
        network (str): VPC network 정보
        input_data (Input[Dataset]): 학습에 사용할 입력 데이터 artifact
        hparams (dict): 하이퍼파라미터
        tensorboard_root (str): tensorboard log 저장 경로 (gs://...)
        train_container_uri (str): training container image URI
        staging_bucket (str): GCS bucket base_output_dir
        train_data, valid_data, test_data (Output[Dataset]): 분할된 출력 dataset
        model (Output[Model]): 학습된 모델 아티팩트
        metrics (Output[Metrics]): 학습 성능 메트릭
        tensorboard_resource_name (str, optional): Tensorboard 리소스 이름
        input_test_path (str, optional): 테스트 데이터 GCS 경로
        job_name_prefix (str): Job 이름 prefix
        replica_count (int): 워커 수
        train_machine (str): 사용할 머신 스펙 ("T4", "L4", "A100", "CPU")

    Returns:
        str: 저장된 모델 경로
    """
    import os
    import json
    import re
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
        },
    }

    if train_machine not in machine_spec:
        train_machine = "T4"

    match = re.match(r"(\d+)(st|nd|rd)", job_name_prefix) # 1st, 2nd, 3rd
    try_prefix = f"-{match.group(0)}" if match else ""

    cmd_args = [
        "--input_path", input_data.path,
        *([f"--input_test_path={input_test_path}"] if input_test_path else []),
        "--hparams", json.dumps(hparams),
        "--output_train_path", train_data.path,
        "--output_valid_path", valid_data.path,
        "--output_test_path", test_data.path,
        "--output_model", model.path,
        "--output_metrics", metrics.path,
        "--timestamp", job_name_suffix + try_prefix,
        # "--tensorboard_root", tensorboard_root  # 자동화된 파이프라인에서는 불필요
    ]

    experiment_name = f"my-experiment-{job_name_suffix}"

    aiplatform.init(
        project=project,
        location=location,
    )

    job = aiplatform.CustomContainerTrainingJob(
        display_name=f"{job_name_prefix}{job_name_suffix}",
        container_uri=train_container_uri,
        script_path="model-training/train.py",
    )

    train_model_job = job.run(
        base_output_dir=staging_bucket,
        service_account=service_account,
        network=network,
        args=cmd_args,
        replica_count=replica_count,
        **machine_spec[train_machine],
        tensorboard=tensorboard_resource_name,
    )

    model.metadata["framework"] = "pytorch"
    model.metadata["file_path"] = model.uri + ".pth"
    model.metadata["hparam"] = json.dumps(hparams)

    for ds in [train_data, valid_data, test_data]:
        ds.metadata["number"] = len(os.listdir(ds.path))
        ds.metadata["data_type"] = "image"

    if os.path.isfile(metrics.path):
        with open(metrics.path, "r") as fp:
            parsed_metrics = json.load(fp)
        for k, v in parsed_metrics.items():
            if isinstance(v, float):
                metrics.log_metric(k, v)

    return model.metadata["file_path"]
