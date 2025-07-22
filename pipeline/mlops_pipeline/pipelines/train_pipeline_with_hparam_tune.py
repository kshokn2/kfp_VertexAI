"""
하이퍼파라미터를 튜닝을 선행하는 학습 파이프라인을 정의합니다.
"""

import os
from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output, component
from get_env import EVAL_PIPELINE_NAME

from components import get_param, data_ingestion, service_evaluate, hyperparam_tune

@dsl.pipeline(name="파이프라인 이름")
def pipeline(
    project: str = os.environ.get("VERTEX_PROJECT_ID"),
    location: str = os.environ.get("VERTEX_LOCATION"),
    # ... ,
    # ... ,
    train_machine: str = "T4",
):

    """
    데이터 인입 및 학습 준비하는 컴포넌트 정의
    """
    param_op = get_param.train_param2( ... )
    data_op = data_ingestion.download_images( ... )


    """
    하이퍼 파라미터 튜닝 컴포넌트 구현 예시.
    ps. 모듈화를 더 한다면 관련 컴포넌트를 만들어서 HyperparameterTuningJobRunOp를 실행하는 방향으로 리팩토링 및 테스트 필요.
    
    ------------------------------------ 예시 코드 ------------------------------------
    from google_cloud_pipeline_components.v1 import hyperparameter_tuning_job
    from google_cloud_pipeline_components.v1 hyperparameter_tuning_job import HyperparameterTuningJobRunOp

    TUNE_HPARAMS = dict(
        hpt_epochs = 10,
        hpt_max_trial_count = 8,
        hpt_parallel_trial_count = 2,
        조정_파라미터1 = myparam1,
    )

    hpt_worker_pool_spec = [{
        "machine_spec": {
            "machine_type": "my-gcp-machine",
            "accelerator_type": "my-gcp-accelerator",
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "container_spec": {"image_uri": hpt_image, "args": ["--epochs", str(TUNE_HPARAMS["hpt_epochs"]), "--조정_파라미터1", TUNE_HPARAMS["myparam1"],]},
    }]
    hpt_study_spec_metrics = hyperparameter_tuning_job.serialize_metrics({"내_메트릭1": "maximize"})
    
    hpt_study_spec_parameters = hyperparameter_tuning_job.serialize_metrics({
        "learning_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(
            min=0.0001, max=0.01, scale="log",
        ),
        # "example_parameter": aiplatform.hyperparameter_tuning.DiscreteParameterSpec 등..
    })

    # hyperparameter tuning
    tuning_op = HyperparameterTuningJobRunOp(
        display_name="hyperparameter-tuning",
        project = project,
        location = location,
        worker_pool_specs = hpt_worker_pool_spec,
        study_spec_metrics = hpt_study_spec_metrics,
        study_spec_parameters = hpt_study_spec_parameters,
        max_trial_count = TUNE_HPARAMS["hpt_max_trial_count"],
        parallel_trial_count = TUNE_HPARAMS["hpt_parallel_trial_count"],
        base_output_directory = param_op.outputs["pipeline_root"],
    ).after(data_op).set_caching_options(False)

    best_trial_op = hyperparam_tune.GetBestTrialOp(
        gcp_resources = tuning_op.outputs["gcp_resources"],
        study_spec_metrics = hpt_study_spec_metrics,
    ).set_display_name("get-best-trial").set_caching_options(False)

    best_param_op = hyperparam_tune.GetHyperparametersOp(
        trial = best_trial_op.output,
    ).set_display_name("get-best-parameters").set_caching_options(False)
    """
    

    """
    학습 및 평가하는 컴포넌트 정의
    best_param_op.output를 사용하여 구현..
    """
    train_op = model_training.custom_train_job2( ... )
    evaluate_op = service_evaluate.evaluate_metric2( ... )


    """
    엔드포인트 배포 및 모델 등록하는 컴포넌트 구현 예시.
    
    ------------------------------------ 예시 코드 ------------------------------------
    `from google_cloud_pipeline_components.types.artifact_types import VertexEndpoint`를 사용
    ps. deploy_model 컴포넌트에서는 VertexEndpoint 아티팩트를 Input으로 받아서,
        aiplatform.Model.upload_tensorflow_saved_model 정의와 그것의 deploy() 매서드로 배포 진행.

    from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp

    # 엔드포인트 생성
    create_endpoint_op = EndpointCreateOp(
        display_name = create_private_ep_name,
        project = project_id,
        location = region,
        network = vpc_network,
    ).after(evaluate_op).set_caching_options(False)

    model_deploy_op = deploy_model( ... )
    """