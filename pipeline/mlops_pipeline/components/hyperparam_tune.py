"""
하이퍼파라미터 조정에 관련된 Custom Kubeflow Component를 정의합니다.
"""

from typing import NamedTuple
from get_env import BASE_IMAGE_URI
from kfp.dsl import component

@component(
    base_image=BASE_IMAGE_URI,
)
def GetBestTrialOp(
    gcp_resources: str,
    study_spec_metrics: list,
) -> str:
    import sys
    import logging

    from google.cloud import aiplatform
    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources
    from google.protobuf.json_format import Parse
    from google.cloud.aiplatform_v1.types import study

    api_endpoint_suffix = '-aiplatform.googleapis.com'
    gcp_resources_proto = Parse(gcp_resources, GcpResources())
    gcp_resources_split = gcp_resources_proto.resources[0].resource_uri.partition('projects')
    resource_name = gcp_resources_split[1] + gcp_resources_split[2]
    prefix_str = gcp_resources_split[0]
    prefix_str = prefix_str[:prefix_str.find(api_endpoint_suffix)]
    api_endpoint = prefix_str[(prefix_str.rfind('//') + 2):] + api_endpoint_suffix

    client_options = {'api_endpoint': api_endpoint}
    job_client = aiplatform.gapic.JobServiceClient(client_options=client_options)
    response = job_client.get_hyperparameter_tuning_job(name=resource_name)
    
    trials = [study.Trial.to_json(trial) for trial in response.trials]

    if len(study_spec_metrics) > 1:
        raise RuntimeError('Unable to determine best parameters for multi-objective hyperparameter tuning.')
        logging.error("Unable to determine best parameters for multi-objective hyperparameter tuning.")
        sys.exit(1)
    trials_list = [study.Trial.from_json(trial) for trial in trials]
    best_trial = None
    goal = study_spec_metrics[0]['goal']
    best_fn = None
    if goal == study.StudySpec.MetricSpec.GoalType.MAXIMIZE:
        best_fn = max
    elif goal == study.StudySpec.MetricSpec.GoalType.MINIMIZE:
        best_fn = min
    best_trial = best_fn(
        trials_list, key=lambda trial: trial.final_measurement.metrics[0].value)
    
    print("GetBestTrialOp component completed..")

    return study.Trial.to_json(best_trial)


@component(
    base_image=BASE_IMAGE_URI,
)
def GetHyperparametersOp(
    trial: str,
) -> list:
    from google.cloud.aiplatform_v1.types import study

    trial_proto = study.Trial.from_json(trial)

    return [ study.Trial.Parameter.to_json(param) for param in trial_proto.parameters ]
