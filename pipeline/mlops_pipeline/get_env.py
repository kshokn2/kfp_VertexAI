"""
project에서의 기본적인 환경 변수(project_id, region 등..)들과
Kubeflow에서의 환경 변수(Base Image URI, 학습/평가 Container URI 등)들을 공통적으로 설정하고 활용합니다.

(참고 링크)
https://github.com/GoogleCloudPlatform/vertex-pipelines-end-to-end-samples/blob/main/pipelines/src/pipelines/trigger/main.py#L166
"""

import os
from dotenv import load_dotenv

from common.gcp_utils import access_secret_version, get_project_num

def get_env() -> dict:
    # Get the necessary environment variables for pipeline runs,
    # and return them as a dictionary.
    if "VERTEX_PROJECT_ID" not in os.environ and "VERTEX_LOCATION" not in os.environ:
        load_dotenv(".env")

    PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "my_project")
    REGION = os.getenv("VERTEX_LOCATION", "asia-northeast3")
    PROJECT_NUM = get_project_num(PROJECT_ID)

    PIPELINE_ROOT = access_secret_version(PROJECT_NUM, "gcs-vertex-pipeline-root", "latest")
    PIPELINEJOB_SA = access_secret_version(PROJECT_NUM, "sa-vertex", "latest")  # "service account email"
    PRIVATE_EP_VPC = access_secret_version(PROJECT_NUM, "vpc-vertex", "latest")
    ENCRYPTION_SPEC_KEY_NAME = None # os.environ.get("VERTEX_CMEK_IDENTIFIER")
    DOCKER_URI = access_secret_version(PROJECT_NUM, "ar-docker", "latest")
    KFP_URI = access_secret_version(PROJECT_NUM, "ar-kfp-uri", "latest")
    BASE_IMAGE = access_secret_version(PROJECT_NUM, "kfp-base-image", "latest")

    return {
        "project_id": PROJECT_ID,
        "location": REGION,
        "project_num": PROJECT_NUM,
        "pipeline_root": PIPELINE_ROOT,
        "service_account": PIPELINEJOB_SA,
        "encryption_spec_key_name": ENCRYPTION_SPEC_KEY_NAME,
        "docker_uri": DOCKER_URI,
        "kfp_uri": KFP_URI,
        "base_image": BASE_IMAGE,
    }


def set_train_env(parameter_values: dict, custom_option=True):
    parameter_values["train_machine"] = "T4"

    # 커스텀 파일 옵션
    if custom_option:
        parameter_values["db_path"] = "gs://some-bucket/folder/my_db.db"

def set_eval_env(parameter_values: dict, custom_option=True):
    parameter_values["eval_machine"] = "A100"

    # 커스텀 파일 옵션
    if custom_option:
        parameter_values["db_path"] = "gs://some-bucket/folder/my_db.db"
        # parameter_values["model_path"] = "gs://some-bucket/folder/model.pth"
        parameter_values["gtfile_path"] = "gs://some-bucket/folder/정답.xlsx"
        parameter_values["model_path"] = "pretrained"

def set_e2e_env(parameter_values: dict, custom_option=True):
    set_train_env(parameter_values, custom_option=False)
    set_eval_env(parameter_values, custom_option=False)

    # 커스텀 파일 옵션
    if custom_option:
        parameter_values["db_path"] = "gs://some-bucket/folder/my_db.db"
        # parameter_values["gtfile_path"] = "gs://some-bucket/folder/정답.xlsx"

env = get_env()
BASE_IMAGE_URI = env["base_image"]
TRAIN_PIPELINE_NAME = f'{env["location"]}-docker.pkg.dev/{env["project_id"]}/{env["docker_uri"]}/training_uri'
EVAL_PIPELINE_NAME = f'{env["location"]}-docker.pkg.dev/{env["project_id"]}/{env["docker_uri"]}/evaluation_uri'