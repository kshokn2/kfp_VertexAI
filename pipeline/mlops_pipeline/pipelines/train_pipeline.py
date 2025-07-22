"""
components에 정의된 기능들을 기반으로 Kubeflow 기반의 학습 파이프라인을 정의합니다.
해당 파이프라인은 GCP의 Artifact Registry에 등록하고, 사용자가 메뉴얼하게 실행할 수 있도록 최소한의 파라미터만으로 설정합니다.

(참고. 파이프라인 함수의 파라미터들에 default 값을 설정하면 GCP Console(Vertex AI > Pipeline > Your Template) 상에서 파라미터가 자동으로 값이 매핑되어 있습니다.)
"""

import os
from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output, component
from get_env import TRAIN_PIPELINE_NAME

from components import get_param, data_ingestion, model_training


@dsl.pipeline(name=TRAIN_PIPELINE_NAME)
def pipeline(
    project: str = os.environ.get("VERTEX_PROJECT_ID"),
    location: str = os.environ.get("VERTEX_LOCATION"),
    timestamp: str = "20250101-0000",
    db_path: str = "prod",  # "gs://bucket/folder/file",
    # test_data_gcs_uri: str = "gs://my_bucket/my_testset_dir",
    train_machine: str = "T4",
):
    task_name = "mlops-pipeline"

    HPARAMS = dict(
        split_ratio=0.8,
        batch_size=64,  # 16
        epochs=50,  # 20
        lr=0.001,
        patience=5,  # validation loss가 개선되지 않을 경우 중단할 epoch 수
    )

    # 학습을 위한 비밀 및 파라미터 가져오는 Component
    param_op = get_param.train_param(
        project=project,
        location=location,
        db_path=db_path,
    ).set_cpu_limit("1").set_memory_limit("2G")

    # 학습을 위한 데이터 다운로드 Component
    data_op = data_ingestion.download_images(
        gcs_db_path=param_op.outputs["db_file_path"],
        table_name=param_op.outputs["db_table_name"],
        bucket_binary_image=param_op.outputs["binary_image_bucket_name"],
        log_msg_prefix=f"[Task #{task_name}]",
        log_fail_date=timestamp,
    ).set_caching_options(True)

    # 모델 학습 Component
    train_op = model_training.custom_train_job(
        project=project,
        location=location,
        job_name_suffix=timestamp,
        service_account=param_op.outputs["service_account"],
        network=param_op.outputs["network"],
        input_data=data_op.outputs["all_img_dir"],
        # input_test_path=test_data_gcs_uri,
        hparams=HPARAMS,
        train_container_uri=param_op.outputs["train_container_uri"],
        staging_bucket=param_op.outputs["pipeline_root"],
        tensorboard_resource_name=param_op.outputs["tensorboard_resource"],
        tensorboard_root="",  # f"gs://{param_op.outputs["pipeline_root"]}/logs"
        machine=train_machine,
    ).set_cpu_limit("1").set_memory_limit("2G")

    return train_op.outputs["Output"]
