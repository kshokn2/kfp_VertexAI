"""
components에 정의된 기능들을 기반으로 Kubeflow 기반의 End To End 파이프라인을 정의합니다. (E2E : [학습+평가]x3 실행)
해당 파이프라인은 GCP의 Artifact Registry에 등록하고, 사용자가 메뉴얼하게 실행할 수 있도록 최소한의 파라미터만으로 설정합니다.

(참고. 파이프라인 함수의 파라미터들에 default 값을 설정하면 GCP Console(Vertex AI > Pipeline > Your Template) 상에서 파라미터가 자동으로 값이 매핑되어 있습니다.)
"""

import os
from typing import NamedTuple
from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output, component
from get_env import E2E_PIPELINE_NAME

from components import get_param, data_ingestion, model_training, service_evaluate, gcp_logger, save_output_json


@dsl.pipeline(name=E2E_PIPELINE_NAME)
def pipeline(
    project: str = os.environ.get("VERTEX_PROJECT_ID"),
    location: str = os.environ.get("VERTEX_LOCATION"),
    timestamp: str = "20250101-0000",
    update_case: str = "biweekly",
    db_path: str = "prod",      # "gs://bucket/folder/file"
    gt_file_path: str = "prod", # "gs://bucket/folder/file"
    train_machine: str = "T4",
    eval_machine: str = "A100",
) -> NamedTuple("pipeline_outputs", [("result", str), ("model_path", str)]):

    task_name = "mlops-pipeline"
    output = NamedTuple("pipeline_outputs", [("result", str), ("model_path", str)])

    HPARAMS = dict(
        split_ratio=0.8,
        batch_size=64,
        epochs=50,
        early_stop_patience=5,
    )

    train_param_op = get_param.train_param(
        project=project, location=location, db_path=db_path
    ).set_cpu_limit("1").set_memory_limit("2G")

    eval_param_op = get_param.eval_param(
        project=project, location=location, db_path=db_path,
        model_path="pass", gt_path=gt_file_path
    ).set_cpu_limit("1").set_memory_limit("2G")

    data_op = data_ingestion.download_images(
        gcs_db_path=train_param_op.outputs["db_file_path"],
        table_name=train_param_op.outputs["db_table_name"],
        bucket_binary_image=train_param_op.outputs["binary_image_bucket_name"],
        log_msg_prefix=f"[Task #{task_name}]",
        log_fail_date=timestamp,
    ).set_caching_options(True)

    model_gt_op = data_ingestion.download_others_for_eval(
        gcs_model_path="pass",
        gcs_gt_path=eval_param_op.outputs["groundtruth_file_path"],
        log_msg_prefix=f"[Task #{task_name}]",
        log_fail_date=timestamp,
    ).set_caching_options(True).after(eval_param_op)

    train_op_try_1st = model_training.custom_train_job(
        project=project,
        location=location,
        job_name_suffix=timestamp,
        service_account=train_param_op.outputs["service_account"],
        network=train_param_op.outputs["network"],
        input_data=data_op.outputs["all_img_dir"],
        hparams=HPARAMS,
        train_container_uri=train_param_op.outputs["train_container_uri"],
        staging_bucket=train_param_op.outputs["pipeline_root"],
        job_name_prefix="1st Try - CustomTrainingJob",
        tensorboard_resource_name=train_param_op.outputs["tensorboard_resource"],
        tensorboard_root="",
        train_machine=train_machine,
    ).set_cpu_limit("1").set_memory_limit("2G")

    evaluate_op_try_1st = service_evaluate.evaluate_metric(
        project=project,
        location=location,
        job_name_suffix=timestamp,
        update_case=update_case,
        input_data=data_op.outputs["all_img_dir"],
        input_db=data_op.outputs["db_dir"],
        input_model=train_op_try_1st.outputs["model"],
        input_groundtruth=model_gt_op.outputs["groundtruth"],
        eval_container_uri=eval_param_op.outputs["eval_container_uri"],
        staging_bucket=eval_param_op.outputs["pipeline_root"],
        job_name_prefix="1st Try - Evaluation_CustomJob",
        eval_machine=eval_machine,
    ).set_cpu_limit("1").set_memory_limit("2G")

    with dsl.If(evaluate_op_try_1st.outputs["Output"] == "rejected", name="1st-evaluation-reject-condition"):
        train_op_try_2nd = model_training.custom_train_job(
            project=project,
            location=location,
            job_name_suffix=timestamp,
            service_account=train_param_op.outputs["service_account"],
            network=train_param_op.outputs["network"],
            input_data=data_op.outputs["all_img_dir"],
            hparams=HPARAMS,
            train_container_uri=train_param_op.outputs["train_container_uri"],
            staging_bucket=train_param_op.outputs["pipeline_root"],
            job_name_prefix="2nd Try - CustomTrainingJob",
            tensorboard_resource_name=train_param_op.outputs["tensorboard_resource"],
            tensorboard_root="",
            train_machine=train_machine,
        ).set_cpu_limit("1").set_memory_limit("2G")

        evaluate_op_try_2nd = service_evaluate.evaluate_metric(
            project=project,
            location=location,
            job_name_suffix=timestamp,
            update_case=update_case,
            input_data=data_op.outputs["all_img_dir"],
            input_db=data_op.outputs["db_dir"],
            input_model=train_op_try_2nd.outputs["model"],
            input_groundtruth=model_gt_op.outputs["groundtruth"],
            eval_container_uri=eval_param_op.outputs["eval_container_uri"],
            staging_bucket=eval_param_op.outputs["pipeline_root"],
            job_name_prefix="2nd Try - Evaluation_CustomJob",
            eval_machine=eval_machine,
        ).set_cpu_limit("1").set_memory_limit("2G")

        with dsl.If(evaluate_op_try_2nd.outputs["Output"] == "rejected", name="2nd-evaluation-reject-condition"):
            train_op_try_3rd = model_training.custom_train_job(
                project=project,
                location=location,
                job_name_suffix=timestamp,
                service_account=train_param_op.outputs["service_account"],
                network=train_param_op.outputs["network"],
                input_data=data_op.outputs["all_img_dir"],
                hparams=HPARAMS,
                train_container_uri=train_param_op.outputs["train_container_uri"],
                staging_bucket=train_param_op.outputs["pipeline_root"],
                job_name_prefix="3rd Try - CustomTrainingJob",
                tensorboard_resource_name=train_param_op.outputs["tensorboard_resource"],
                tensorboard_root="",
                train_machine=train_machine,
            ).set_cpu_limit("1").set_memory_limit("2G")

            evaluate_op_try_3rd = service_evaluate.evaluate_metric(
                project=project,
                location=location,
                job_name_suffix=timestamp,
                update_case=update_case,
                input_data=data_op.outputs["all_img_dir"],
                input_db=data_op.outputs["db_dir"],
                input_model=train_op_try_3rd.outputs["model"],
                input_groundtruth=model_gt_op.outputs["groundtruth"],
                eval_container_uri=eval_param_op.outputs["eval_container_uri"],
                staging_bucket=eval_param_op.outputs["pipeline_root"],
                job_name_prefix="3rd Try - Evaluation_CustomJob",
                eval_machine=eval_machine,
            ).set_cpu_limit("1").set_memory_limit("2G")

            final_result = evaluate_op_try_3rd.outputs["Output"]
            final_model_path = train_op_try_3rd.outputs["Output"]

            with dsl.If(evaluate_op_try_3rd.outputs["Output"] == "rejected", name="3rd-evaluation-reject-condition"):
                gcp_logger.mlops_msg_log(msg=f"[Task #{task_name}] finished. (3차 평가 결과 및 모델 배포: 중단)", severity="WARNING").set_cpu_limit("1").set_memory_limit("2G")
            with dsl.Else(name="3rd-evaluation-approval-condition"):
                gcp_logger.mlops_msg_log(msg=f"[Task #{task_name}] finished. (3차 평가 결과 및 모델 배포: 승인)", severity="INFO").set_cpu_limit("1").set_memory_limit("2G")

        with dsl.Else(name="2nd-evaluation-approval-condition"):
            gcp_logger.mlops_msg_log(msg=f"[Task #{task_name}] finished. (2차 평가 결과 및 모델 배포: 승인)", severity="INFO").set_cpu_limit("1").set_memory_limit("2G")
            final_result = evaluate_op_try_2nd.outputs["Output"]
            final_model_path = train_op_try_2nd.outputs["Output"]

    with dsl.Else(name="1st-evaluation-approval-condition"):
        gcp_logger.mlops_msg_log(msg=f"[Task #{task_name}] finished. (1차 평가 결과 및 모델 배포: 승인)", severity="INFO").set_cpu_limit("1").set_memory_limit("2G")
        final_result = evaluate_op_try_1st.outputs["Output"]
        final_model_path = train_op_try_1st.outputs["Output"]

    save_output_json.save_e2e_output(
        deploy_result=final_result,
        deploy_model=final_model_path,
        eval_artifacts=evaluate_op_try_1st.outputs["deployment_status"],
    ).set_cpu_limit("1").set_memory_limit("2G")

    return output(final_result, final_model_path)
