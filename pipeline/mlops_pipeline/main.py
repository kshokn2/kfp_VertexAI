"""
GCP에서 kubeflow pipeline을 Artifact Registry로 uplaod 하거나 Pipeline Job을 trigger하는 main 함수

(참고 링크)
https://github.com/GoogleCloudPlatform/vertex-pipelines-end-to-end-samples/blob/main/pipelines/src/pipelines/trigger/main.py
https://github.com/teamdatatonic/vertex-pipelines-end-to-end-samples/blob/develop/pipelines/src/pipelines/utils/trigger_pipeline.py
"""

import os
import sys
import argparse
import traceback
from datetime import datetime
import pytz
from typing import Optional, List

from google.cloud import aiplatform #, logging
from kfp import compiler

from get_env import *
from pipelines import train_pipeline, evaluate_pipeline, e2e_pipeline
# from pipelines_test import pipeline_test1, pipeline_test2, pipeline_test3
# from pipelines import pipeline_test5 # logging test pipe


# logging 설정
# log_name = "my_logname"
# logging_client = logging.Client()
# logger = logging_client.logger(log_name)

allow_tasks = ["e2e", "train", "evaluate"] # + ["test1", "test2", "test3", "test5"]

def trigger_pipeline_from_payload(payload: dict) -> aiplatform.PipelineJob:
    return trigger_pipeline(
        project_id=env["project_id"],
        location=env["location"],
        project_num=env["project_num"],
        template_path=payload["attributes"]["template_path"],
        pipeline_name=f'{env["project_id"]}-{payload["attributes"]["exec_task"]}-pipeline',
        pipeline_root=env["pipeline_root"],
        parameter_values=payload["data"],
        service_account=env["service_account"],
        network=env["network"],
        exec_task=payload["attributes"]["exec_task"],
        encryption_spec_key_name=env["encryption_spec_key_name"],
        enable_caching=payload["attributes"]["enable_caching"],
    )

def trigger_pipeline(
    project_id: str,
    location: str,
    project_num: str,
    template_path: str,
    pipeline_name: str,
    pipeline_root: str,
    parameter_values: dict,
    service_account: str,
    network: str,
    exec_task: str,
    encryption_spec_key_name: Optional[str] = None,
    enable_caching: Optional[bool] = False,
):
    kst = pytz.timezone("Asia/Seoul")

    parameter_values["project"] = project_id
    parameter_values["location"] = location
    parameter_values["timestamp"] = datetime.now(kst).strftime("%Y%m%d-%H%M")

    # Initialise API client
    aiplatform.init(project=project_id, location=location)

    if exec_task == "train":
        set_train_env(parameter_values, custom_option=False)
    elif exec_task == "evaluate":
        set_eval_env(parameter_values, custom_option=False)
    elif exec_task == "e2e":
        set_e2e_env(parameter_values, custom_option=False)

    print(f"입력 파라미터 : {list(parameter_values.keys())}")

    # Instantiate PipelineJob object
    job = aiplatform.PipelineJob(
        display_name="my-mlops-pipeline",
        template_path=template_path,
        job_id=f'{pipeline_name}-{parameter_values["timestamp"]}',
        pipeline_root=pipeline_root,
        parameter_values=parameter_values,
        encryption_spec_key_name=encryption_spec_key_name,
        enable_caching=enable_caching,
    )

    # Submit: aiplatform.PipelineJob
    job.submit(
        service_account=service_account,
        network=network,
    )

    return job

def upload_pipeline_from_yaml(pipeline_path: str):
    """
    Upload the pipeline to Artifact Registry.
    """
    from kfp.registry import RegistryClient

    host = f'https://{env["location"]}-kfp.pkg.dev/{env["project_id"]}/{env["kfp-uri"]}'
    client = RegistryClient(host=host)
    try:
        client.upload_pipeline(
            file_name=pipeline_path,
            tags=["latest"],
        )
        print(f"upload to artifact registry : {host}")
    except Exception as e:
        print("Failed to upload kfp to artifact registry.\n")
        print(e)

    return pipeline_path

def main(args):
    print("argument : ", args)

    payload = {
        "attributes": {
            "run_type": args.run_type,
            "exec_task": args.exec_task,
            "template_path": args.template_path,
            "enable_caching": args.enable_caching,
        },
        "data": {"update_case": "biweekly"},
    }

    template_dir = os.path.dirname(args.template_path)

    # 템플릿 저장할 directory 생성
    os.makedirs(template_dir, exist_ok=True)

    if os.path.isfile(args.template_path):
        os.remove(args.template_path)
        print("이전 yaml 파일 삭제 완료.")

    # payload의 data(추가 파라미터) 부분 완성
    if args.exec_task == "e2e":
        compiler.Compiler().compile(
            pipeline_func=e2e_pipeline.pipeline,
            package_path=args.template_path
        )
    elif args.exec_task == "train":
        compiler.Compiler().compile(
            pipeline_func=train_pipeline.pipeline,
            package_path=args.template_path
        )
    elif args.exec_task == "evaluate":
        compiler.Compiler().compile(
            pipeline_func=evaluate_pipeline.pipeline,
            package_path=args.template_path
        )
    # elif args.exec_task == "test1":
    #     # "이건 테스트용이므로 테스트 후 삭제"
    #     compiler.Compiler().compile(
    #         pipeline_func=pipeline_test1.pipeline,
    #         package_path=args.template_path
    #     )
    # elif args.exec_task == "test2":
    #     # "이건 테스트용이므로 테스트 후 삭제"
    #     compiler.Compiler().compile(
    #         pipeline_func=pipeline_test2.pipeline,
    #         package_path=args.template_path
    #     )
    # elif args.exec_task == "test3":
    #     # "이건 테스트용이므로 테스트 후 삭제"
    #     compiler.Compiler().compile(
    #         pipeline_func=pipeline_test3.pipeline,
    #         package_path=args.template_path
    #     )
    else:
        print("argument 에러 발생")
        msg = f"입력된 Argument의 exec_task인 {args.exec_task}에 대해서 기능 구현되어있지 않습니다."
        raise RuntimeError(msg)

    print(f"Pipeline 코드 컴파일 완료.\n{args.template_path} 파일 생성 완료.")


# run_type에 맞게 실행
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--run_type", help="Choose 'trigger' or 'upload' for a purpose. Default value is 'upload'", type=str, default="upload")
        parser.add_argument("--exec_task", help=f"Choose the purpose of executing the pipeline {allow_tasks}", choices=allow_tasks, type=str, default="e2e")
        parser.add_argument("--template_path", help="Path to the compiled pipeline (JSON)", type=str, default="./kfp_pipeline.yaml")
        parser.add_argument("--enable_caching", type=bool, default=False)

        # Get commandline args
        args = parser.parse_args()

        task_name = f"kfp-{args.exec_task}-{args.run_type}"
        print(f"[Task #{task_name}] start")
        main(args)
        print(f"[Task #{task_name}] finished.")

    except Exception as err:
        print(traceback.format_exc())
        print(f"[Task #{task_name}] failed. err: {str(err)}")
        sys.exit(1)