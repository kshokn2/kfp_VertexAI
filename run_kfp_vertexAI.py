import os
import kfp
from kfp import dsl, compiler
from kfp.dsl import (Artifact, Dataset, Input, Output, Model, Metrics, Markdown, HTML, component, InputPath, OutputPath, PipelineTaskFinalStatus)

from google_cloud_pipeline_components.types.artifact_types import VertexEndpoint
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp
from google_cloud_pipeline_components.v1.custom_job import utils
from google_cloud_pipeline_components.v1 import hyperparameter_tuning_job
from google_cloud_pipeline_components.v1 hyperparameter_tuning_job import HyperparameterTuningJobRunOp

from google.cloud import aiplatform

myparam1 = os.getenv("param1")

# 단위 테스트용
# myparam1 = "parameter1"

PROJECT_ID = "my-project-id"
PROJECT_NUM = "1234567890"
REGION = "my-region"
PIPELINEJOB_SA = "my-service-account@~~~~~"
PRIVATE_EP_VPC = "my-vpc-name"

PIPELINE_ROOT = "gs://my-mlops-bucket"
PIPELINE_NAME = "my-pipeline-name"

BASE_IMAGE = "my-docker-kfp_2_7_0-image" # kfp 2.7.0 설치된 기본 도커 이미지

def main():

    @component(
        base_image=BASE_IMAGE,
        # packages_to_install=["install_library1==0.0.1", "install_library2==1.0.0"]
    )
    def download_data(
        # project: str,
        # location: str,
        bukcet_name: str,
        download_path: OutputPath("Any"),
    ):
        import os
        import sys
        import logging

        # from google.cloud import storage
        from google.cloud import logging as gc_logging

        def download_bucket_with_transfer_manager(
            bucket_name, blob_name_prefix, target_list=None, destination_directory="", workers=8
        ):
            # https://cloud.google.com/storage/docs/samples/storage-transfer-manager-download-bucket?hl=ko

            import os
            import time
            from google.cloud.storage import Client, transfer_manager

            storage_client = Client()
            bucket = storage_client.bucket(bucket_name)

            if target_list is None:
                blob_names = [blob.name.replace(blob_name_prefix, '') for blob in bucket.list_blobs(prefix=blob_name_prefix) if blob.name != blob_name_prefix]
            else:
                blob_names = target_list

            results = transfer_manager.download_many_to_path(
                bucket, blob_names, destination_directory=destination_directory, blob_name_prefix=blob_name_prefix, max_workers=workers)
            
            retry_blobs = []

            for name, result in zip(blob_names, results):
                if isinstance(result, Exception):
                    # print("Failed to download {} due to exception: {}".format(name, result))
                    retry_blobs.append(name)
                # else:
                #     print("Downloaded {} to {}.".format(name, destination_directory + name))

            if len(retry_blobs) != 0:
                retry_results = transfer_manager.download_many_to_path(
                    bucket, retry_blobs, destination_directory=destination_directory, blob_name_prefix=blob_name_prefix, max_workers=workers)
                
                for name, result in zip(retry_blobs, retry_results):
                    if isinstance(result, Exception):
                        logging.error("Failed to download {} due to exception: {}".format(name, result))
                        os.remove(f'{destination_directory}/{name}')
                        time.sleep(0.05)

        workers = int(os.cpu_count()/2) if os.cpu_count() > 61 else os.cpu_count()
        os.makedirs(download_path, exist_ok=True)

        download_bucket_with_transfer_manager(
            bucket_name=bukcet_name,
            blob_name_prefix=f'my_blob/',
            destination_directory=download_path,
            workers=workers
        )

        if len(os.listdir(download_path)) == 0:
            logging.error("download_data component failed..")
            sys.exit(1)

        logging.info("download_data component completed..")


    @component(
        base_image=BASE_IMAGE,
        # packages_to_install=["install_library1==0.0.1", "install_library2==1.0.0"]
    )
    def data_preparation(
        # project: str,
        # location: str,
        rand_seed: int,
        ratio: float,
        batch_size: int,
        label_files: str,
        download_path: InputPath("Any"),
        trainset_path: OutputPath("Any"),
        validset_path: OutputPath("Any"),
        testset_path: OutputPath("Any"),
    ):
        import os
        import sys
        import random
        import numpy as np
        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        import logging

        from google.cloud import storage
        from google.cloud import logging as gc_logging

        random.seed(rand_seed)

        file_list = os.listdir(download_path)
        """
        label 정보 가져오기
        """

        array_data = np.array([np.load(os.path.join(download_path, filenm+".npy")) for filenm in file_list])
        array_class = label_files#로 만들기

        # split dataset
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            array_data, array_class, test_size=1-ratio, random_state=rand_seed, stratify=array_class)
        
        rest_ratio = 0.5
        valid_data, test_data, valid_labels, test_labels = train_test_split(
            temp_data, temp_labels, test_size=rest_ratio, random_state=rand_seed, stratify=temp_labels)
        
        # tf.data.Dataset 생성
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        train_dataset = train_dataset.shuffle(00).batch(batch_size)
        
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_data, valid_labels))
        valid_dataset = valid_dataset.shuffle(00).batch(batch_size)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
        test_dataset = test_dataset.shuffle(00).batch(batch_size)

        # artifacts의 metadata에 저장
        train_dataset.save(trainset_path)
        valid_dataset.save(validset_path)
        test_dataset.save(testset_path)

        logging.info("data_preparation component completed..")


    @component(
        base_image=BASE_IMAGE,
    )
    def worker_pool_spec(
        hpt_epochs: int,
        hpt_image: str,
        myparam1: str,
    ) -> list:
        CMDARGS = [
            "--epochs", str(hpt_epochs),
            "--param1", myparam1,
        ]

        worker_pool_spec = [
            {
                "machine_spec": {
                    # gpu
                    "machine_type": "my-gcp-machine",
                    "accelerator_type": "my-gcp-accelerator",
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "container_spec": {"image_uri": hpt_image, "args": CMDARGS},
            }
        ]

        logging.info("worker_pool_spec component completed..")

        return worker_pool_spec


    @component(
        base_image=BASE_IMAGE,
        # packages_to_install=["google-cloud-aiplatform", "google-cloud-pipeline-components", "protobuf"],
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
        from google.cloud import logging as gc_logging

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
        
        logging.info("GetBestTrialOp component completed..")

        return study.Trial.to_json(best_trial)


    @component(
        base_image=BASE_IMAGE,
    )
    def GetHyperparametersOp(
        trial: str,
    ) -> list:
        from google.cloud.aiplatform_v1.types import study

        trial_proto = study.Trial.from_json(trial)

        return [ study.Trial.Parameter.to_json(param) for param in trial_proto.parameters ]
    

    @component(
        base_image=BASE_IMAGE,
    )
    def training_model(
        rand_seed: int,
        training_params: dict,
        best_params: list,
        tensorboard_root: str,
        dataset_train: InputPath("Any"),
        dataset_valid: InputPath("Any"),
        dataset_test: InputPath("Any"),
        model: Output[Model],
    ):
        import os
        import sys
        import json
        import random
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from sklearn.utils import class_weight

        from google.cloud import logging as gc_logging

        import logging

        random.seed(rand_seed)
        learning_rate = json.loads(best_params[0])["value"] # format(parma 1개일 때): [{"parameterId": "learning_rate", "value": 0.00123}]

        train_dataset = tf.data.Dataset.load(dataset_train)
        valid_dataset = tf.data.Dataset.load(dataset_valid)
        test_dataset = tf.data.Dataset.load(dataset_test)

        os.environ['AIP_TENSORBOARD_LOG_DIR'] = tensorboard_root
        gpus = tf.config.experimental.list_physical_devices('GPU')

        # 모델 정의
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                new_model = create_model()
        else:
            new_model = create_model()

        # 모델 컴파일
        new_model.complie(
            loss = loss설정,
            optimizer = opt설정(learning_rate=learning_rate),
            metrics = [메트릭1, 메트릭2]
        )

        # 모델 학습
        new_model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs = training_params["epochs"],
            verbose = 1,
            # class_weight=class_weight,
            callbacks = [
                tf.keras.callbacks.Tensorboard(log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR'], histogram_freq=1),
                keras.callbacks.EarlyStopping(설정),
                keras.callbacks.ModelCheckpoint(filepath=model.path, monitor=메트릭1, mode='max', save_best_only=True, save_wweights_only=True)
            ]
        )

        logging.info("training_model component completed..")


    @component(
        base_image=BASE_IMAGE,
    )
    def evaluate_model(
        # project: str,
        # location: str,
        dataset_test: InputPath("Any"),
        model: Input[Model],
    ) -> str:
        from google.cloud import logging as gc_logging

        import tensorflow as tf
        import logging

        """
        모델 정의 코드
        """

        load_model = create_model()
        load_model.load_weights(model.path)

        """
        모델 평가 코드
        """

        logging.info("evaluate_model component completed..")

        #평가 통과
        if contidion:
            logging.info("Model evaluation criteria satisfied.")
            return "true" # 배포
        
        else:
            logging.info("Model evaluation criteria not satisfied.")
            return "false" # 재학습


    @component(
        base_image=BASE_IMAGE,
    )
    def deploy_model(
        project: str,
        location: str,
        model_name: str,
        model: Input[Model],
        endpoint: Input[VertexEndpoint],
        upload_model: Output[Model],
        deploy_model: Output[Model],
    ):
        from google.cloud import aiplatform
        from google.cloud import logging as gc_logging

        import logging

        model_list = aiplatform.Model.list(
            filter=f'display_name="{model_name}"')
        
        if len(model_list) > 0:
            # 등록된 모델에 새 버전 생성
            my_model = model_list[0]
            logging.info(f'Name of uploaded model: {my_model.resource_name}')

            uploaded_model = aiplatform.Model.upload_tensorflow_saved_model(
                saved_model_dir = model.path,
                tensorflow_version = "tf 버전",
                use_gpu = False,
                parent_model = my_model.resource_name,
                is_default_version = True,
                version_description = f"등록할 모델 설명",
            )
        else:
            # 새로운 모델 이름으로 등록
            logging.info(f'Name of New uploaded model: {my_model.resource_name}')

            uploaded_model = aiplatform.Model.upload_tensorflow_saved_model(
                saved_model_dir = model.path,
                tensorflow_version = "tf 버전",
                use_gpu = False,
                parent_model = model_name,
                is_default_version = True,
                version_description = f"등록할 모델 설명",
            )

        created_private_ep = None
        for i in aiplatform.PrivateEndpoint.list():
            if i.resource_name == endpoint.metadata['resourceName']:
                created_private_ep = i
                break

        if created_private_ep:
            # 엔드포인트 재생성 필요
            pass
        else:
            # 앞에서 생성한 ep에 모델 배포
            deployed_model = uploaded_model.deploy(
                endpoint = created_private_ep,
                machine_type = "my-machine",
                # deployed_model_display_name = "my-display-name" # 필요시
            )

        upload_model.uri = uploaded_model.resource_name
        deploy_model.uri = deployed_model.resource_name


        logging.info("deploy_model component completed..")


    @component(
        base_image=BASE_IMAGE,
    )
    def get_run_state(status: dict) -> str:
        return status['state']
        

    @component(
        base_image=BASE_IMAGE,
    )
    def logging_op(msg: str, severity: str):
        from google.cloud import logging as gc_logging
        import logging

        if severity == "WARNING":
            logging.warn(msg)
        elif severity == "ERROR":
            logging.error(msg)
        else:
            logging.info(msg)


    @dsl.pipeline(name='conditional-notification')
    def exit_op(status: PipelineTaskFinalStatus):
        with dsl.If(get_run_state(status=status).output == "FAILED"):
            # clearnup_op() # 파이프라인 실패하면 clearnup
            logging_op(msg="실패 메시지", severity="ERROR").set_display_name("fail-alarm")
        
        with dsl.Else():
            logging_op(msg="성공 메시지", severity="INFO").set_display_name("success-alarm")

    
    @dsl.pipeline(
        name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT
    )
    def pipeline(
        project_id: str,
        project_num: str,
        region: str,
        vpc_network: str,
        param1: str, # hpt에 사용하는 파라미터 예시
        tensorboard_root_try1: str,
        create_private_ep_name: str,
        upload_model_name: str,
    ):
        """
        각종 필요한 파라미터들 설정..
        """
        rand_seed = 42
        download_bukcet_name = "my-bucket"
        hpt_container_image = "hpyerparamter tuning image"
        hpt_epochs = 10
        hpt_max_trial_count = 8
        hpt_parallel_trial_count = 2

        tr_label_files = "my-training-label-file"
        tensorboard_id = "1234567890" # tensorboard id in gcp
        sa_custom_training = "my-service-account"

        training_params = {
            "batch_size": 64,
            "ratio": 0.8,
            "epochs": 1000
        }

        # Data download
        data_download_op = download_data(
            # project = project_id,
            # location = region,
            bukcet_name = download_bukcet_name,
        ).set_display_name("data-download").set_caching_options(False)

        # Data preparation
        preparation_op = data_preparation(
            # project = project_id,
            # location = region,
            rand_seed = rand_seed,
            ratio = training_params["ratio"],
            batch_size = training_params["batch_size"],
            label_files = tr_label_files,
            download_path = data_download_op.outputs["download_path"],
        ).set_display_name("data-preparation").set_cpu_limit('8').set_memory_limit('16G').set_caching_options(False)

        # worker_pool_specs, study_spec_metrics, study_spec_parameters
        hpt_worker_pool_spec = worker_pool_spec(
            hpt_epochs = hpt_epochs,
            hpt_image = hpt_container_image,
            myparam1 = param1,
        ).set_caching_options(False)

        hpt_study_spec_metrics = hyperparameter_tuning_job.serialize_metrics({"메트릭1": "maximize"})
        
        hpt_study_spec_parameters = hyperparameter_tuning_job.serialize_metrics(
            {
                "learning_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(
                    min=0.0001, max=0.01, scale="log",
                ),
                # aiplatform.hyperparameter_tuning.DiscreteParameterSpec
            }
        )

        # hyperparameter tuning
        tuning_op = HyperparameterTuningJobRunOp(
            display_name="hyperparameter-tuning",
            project = project_id,
            location = region,
            worker_pool_specs = hpt_worker_pool_spec.output,
            study_spec_metrics = hpt_study_spec_metrics,
            study_spec_parameters = hpt_study_spec_parameters,
            max_trial_count = hpt_max_trial_count,
            parallel_trial_count = hpt_parallel_trial_count,
            base_output_directory = PIPELINE_ROOT,
        ).after(preparation_op).set_caching_options(False)

        best_trial_op = GetBestTrialOp(
            gcp_resources = tuning_op.outputs["gcp_resources"],
            study_spec_metrics = hpt_study_spec_metrics,
        ).set_display_name("get-best-trial").set_caching_options(False)

        best_param_op = GetHyperparametersOp(
            trial = best_trial_op.output,
        ).set_display_name("get-best-parameters").set_caching_options(False)

        tensorboard = aiplatform.Tensorboard(
            tensorboard_name = tensorboard_id,
            project = "my-prj-id", # 변수로 받으면 안됌
            location = "my-region", # 변수로 받으면 안됌
        )

        # training
        custom_job_training_op = utils.create_custom_training_job_op_from_component(
            training_model,
            tensorboard = tensorboard.resource_name,
            base_output_directory = PIPELINE_ROOT,
            service_account = sa_custom_training,
            # gpu
            machine_type = "my-gcp-machine",
            accelerator_type = "my-gcp-accelerator",
            accelerator_count = "1",
            replica_count = 1
        )
        training_model_op = custom_job_training_op(
            project = project_id, # 필수
            location = region, # 필수
            rand_seed = rand_seed,
            training_params = training_params,
            best_params = best_param_op.output, # ['{"parameterId":"learning_rate","value":0.0001}']
            tensorboard_root = tensorboard_root_try1, # f-string 등으로 조합하면 안됌
            dataset_train = preparation_op.outputs["trainset_path"],
            dataset_valid = preparation_op.outputs["validset_path"],
            dataset_test = preparation_op.outputs["testset_path"],
        ).set_display_name("train-model").set_retry(num_retries=20, backoff_duration="60s", backoff_factor=2, backoff_max_duration="3600s").set_caching_options(False)

        evaluate_op = evaluate_model(
            # project = project_id,
            # location = region,
            dataset_test = preparation_op.outputs["testset_path"],
            model = training_model_op.outputs["model"],
        ).set_display_name("model-evaluation").set_cpu_limit('8').set_memory_limit('16G').set_caching_options(False)

        # 재학습
        with kfp.dsl.If(evaluate_op.output == "false"):
            new_seed = int(time.time())
            print(f'Generate new seed. (value:{new_seed})')

            """
            재학습하는 컴포넌트(또는 위의 컴포넌트들 재정의)
            """

            retry_evaluate_op = evaluate_model(
                # project = project_id,
                # location = region,
                dataset_test = retry_preparation_op.outputs["testset_path"],
                model = retry_training_model_op.outputs["model"],
            ).set_display_name("model-evaluation").set_cpu_limit('8').set_memory_limit('16G').set_caching_options(False)

            # 재학습 성공으로 배포
            with kfp.dsl.If(retry_evaluate_op.output == "true"):
                """
                엔드포인트 생성 및 배포하는 컴포넌트
                """

            # 재학습 실패로 파이프라인 종료
            with kfp.dsl.Else():
                logging_op(msg="재학습 실패 메시지", severity="ERROR").set_display_name("retraining-fail-alert").set_caching_options(False)


        with kfp.dsl.Else():
            # 엔드포인트 생성
            create_endpoint_op = EndpointCreateOp(
                display_name = create_private_ep_name,
                project = project_id,
                location = region,
                network = vpc_network,
            ).after(evaluate_op).set_display_name("create-private-endpoint").set_caching_options(False)

            model_deploy_op = deploy_model(
                project = project_id,
                location = region,
                model_name = upload_model_name,
                model = training_model_op.outputs["model"],
                endpoint = create_endpoint_op.outputs["endpoint"],
            ).set_display_name("deploy-model").set_caching_options(False)

    @dsl.pipeline(name='kfp-pipeline-exit-handler')
    def pipeline_exit_handler(
        project_id: str,
        project_num: str,
        region: str,
        vpc_network: str,
        param1: str, # hpt에 사용하는 파라미터 예시
        tensorboard_root_try1: str,
        create_private_ep_name: str,
        upload_model_name: str,
    ):
        exit_task = exit_op()

        with dsl.ExitHandler(exit_task):
            pipeline(
                project_id = PROJECT_ID,
                project_num = PROJECT_NUM,
                region = REGION,
                vpc_network = PRIVATE_EP_VPC,
                param1 = myparam1,
                tensorboard_root_try1 = tensorboard_root_try1,
                create_private_ep_name = create_private_ep_name,
                upload_model_name = upload_model_name,
            )

    yaml_file = "./kfp_pipeline.yaml"

    compiler.Compiler().compile(
        pipeline_func = pipeline_exit_handler,
        package_path = yaml_file
    )

    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.PipelineJob(
        display_name = PIPELINE_NAME,
        template_path = yaml_file,
        pipeline_root = PIPELINE_ROOT,
        parameter_values = {
            "project_id": PROJECT_ID,
            "project_num": PROJECT_NUM,
            "region": REGION,
            "vpc_network": PRIVATE_EP_VPC,
            "param1": myparam1, # cloud run으로 넘겨받는 파라미터1
            "tensorboard_root_try1": f"{PIPELINE_ROOT}/experiment/tensorboard_log/try1/",
            "create_private_ep_name": f"my-private-endpoint-{241231}",
            "upload_model_name": f"my-upload-model-{1}",
        }
    )

    job.submit(
        service_account = PIPELINEJOB_SA,
        network = PRIVATE_EP_VPC,
    )

if __name__ == "__main__":
    main()