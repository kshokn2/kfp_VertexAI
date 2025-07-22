"""
components에 정의된 기능들을 기반으로 Kubeflow 기반의 평가 파이프라인을 정의합니다.
해당 파이프라인은 GCP의 Artifact Registry에 등록하고, 사용자가 메뉴얼하게 실행할 수 있도록 최소한의 파라미터만으로 설정합니다.

(참고. 파이프라인 함수의 파라미터들에 default 값을 설정하면 GCP Console(Vertex AI > Pipeline > Your Template) 상에서 파라미터가 자동으로 값이 매핑되어 있습니다.)
"""

import os
from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output, component
from get_env import EVAL_PIPELINE_NAME

from components import get_param, data_ingestion, service_evaluate


@dsl.pipeline(name=EVAL_PIPELINE_NAME)
def pipeline(
    project: str = os.environ.get("VERTEX_PROJECT_ID"),
    location: str = os.environ.get("VERTEX_LOCATION"),
    timestamp: str = "20250101-0000",
    update_case: str = "biweekly",
    db_path: str = "prod",  # "gs://bucket/folder/file"
    model_path: str = "prod",  # "pretrained" or "gs://bucket/folder/file"
    gt_file_path: str = "prod",  # "gs://bucket/folder/file"
    eval_machine: str = "A100",
) -> str:

    task_name = "mlops-pipeline"

    # 평가를 위한 비밀 및 파라미터 가져오는 Component
    param_op = get_param.eval_param(
        project=project,
        location=location,
        db_path=db_path,
        model_path=model_path,
        gt_path=gt_file_path,
    ).set_cpu_limit("1").set_memory_limit("2G")

    # 평가를 위한 데이터 다운로드 Component 1
    data_op = data_ingestion.download_images(
        gcs_db_path=param_op.outputs["db_file_path"],
        table_name=param_op.outputs["db_table_name"],
        bucket_binary_image=param_op.outputs["binary_image_bucket_name"],
        log_msg_prefix=f"[Task #{task_name}]",
        log_fail_date=timestamp,
    ).set_caching_options(True)

    # 평가를 위한 데이터 다운로드 Component 2
    model_gt_op = data_ingestion.download_others_for_eval(
        gcs_model_path=param_op.outputs["model_file_path"],
        gcs_gt_path=param_op.outputs["groundtruth_file_path"],
        log_msg_prefix=f"[Task #{task_name}]",
        log_fail_date=timestamp,
    ).set_caching_options(True)

    # 모델 평가 Component
    evaluate_op = service_evaluate.evaluate_metric(
        project=project,
        location=location,
        job_name_suffix=timestamp,
        update_case=update_case,
        input_data=data_op.outputs["all_img_dir"],
        input_db=data_op.outputs["db_dir"],
        input_model=model_gt_op.outputs["model"],
        input_groundtruth=model_gt_op.outputs["groundtruth"],
        eval_container_uri=param_op.outputs["eval_container_uri"],
        staging_bucket=param_op.outputs["pipeline_root"],
        # job_name_prefix="Custom JobName - Evaluation_CustomJob",
        eval_machine=eval_machine,
    ).set_cpu_limit("1").set_memory_limit("2G")

    return evaluate_op.outputs["Output"]
