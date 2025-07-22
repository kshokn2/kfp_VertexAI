"""
파이프라인에서의 파라미터 설정에 관련된 Custom Kubeflow Component를 정의합니다.
"""

from typing import NamedTuple
from get_env import BASE_IMAGE_URI
from kfp.dsl import component

@component(
    base_image=BASE_IMAGE_URI,
)
def train_param(
    project: str,
    location: str,
    db_path: str,
) -> NamedTuple('parameters', pipeline_root=str, service_account=str, network=str, db_file_path=str, db_table_name=str, binary_image_bucket_name=str, train_container_uri=str, tensorboard_resource=str):
    import sys
    from google.cloud import storage, secretmanager
    from googleapiclient import discovery
    from google.auth import default
    import google_crc32c

    def get_project_num(project_id: str) -> str:
        credentials, _ = default()
        service = discovery.build('cloudresourcemanager', 'v1', credentials=credentials)
        request = service.projects().get(projectId=project_id)
        response = request.execute()
        return response['projectNumber']

    def access_secret_version(project_id: str, secret_id: str, version_id: str):
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        try:
            response = client.access_secret_version(request={"name": name})
        except Exception as e:
            print(e)
            return None
        crc32c = google_crc32c.Checksum()
        crc32c.update(response.payload.data)
        if response.payload.data_crc32c != int(crc32c.hexdigest(), 16):
            print("Data corruption detected.")
            return None
        payload = response.payload.data.decode("utf-8")
        return payload

    def get_tb_resource(project_id: str, location: str, tb_id: str) -> str:
        from google.cloud import aiplatform
        if any(tb_id in tb.resource_name for tb in aiplatform.Tensorboard.list()):
            tb = aiplatform.Tensorboard(tensorboard_name=tb_id, project=project_id, location=location)
            return tb.resource_name
        else:
            return None

    project_num = get_project_num(project)
    pipeline_root = access_secret_version(project_num, "gcs-vertex-pipeline-root", "latest")
    service_account = access_secret_version(project_num, "sa-vertex", "latest")
    network = access_secret_version(project_num, "vpc-network", "latest")
    tensorboard_id = access_secret_version(project_num, "vai-tensorboard-id", "latest")
    tensorboard_resource = get_tb_resource(project, location, tb_id=tensorboard_id)
    docker_uri = access_secret_version(project_num, "ar-docker", "latest")
    train_container_uri = access_secret_version(project_num, "container-train-uri", "latest")
    table_name = access_secret_version(project_num, "image-info-table-name", "latest")

    if db_path == "prod":
        db_bucket = access_secret_version(project_num, "gcs-backup", "latest")
        db_file_name = access_secret_version(project_num, "image-info-db-filename", "latest")
        db_path = f"gs://{db_bucket}/{db_file_name}"
    else:
        if not db_path.startswith("gs://"):
            print(f"Invalid GCS URI({db_path}). It should start with 'gs://'")
            sys.exit(1)

    binary_image_bucket = access_secret_version(project_num, "gcs-storage-binary-image", "latest")

    outputs = NamedTuple('parameters', [
        ('pipeline_root', str),
        ('service_account', str),
        ('network', str),
        ('db_file_path', str),
        ('db_table_name', str),
        ('binary_image_bucket_name', str),
        ('train_container_uri', str),
        ('tensorboard_resource', str),
    ])
    return outputs(
        pipeline_root,
        service_account,
        f'projects/{project_num}/global/networks/{network}',
        db_path,
        table_name,
        binary_image_bucket,
        f'{location}-docker.pkg.dev/{project}/{docker_uri}/{train_container_uri}',
        tensorboard_resource,
    )


@component(
    base_image=BASE_IMAGE_URI,
)
def eval_param(
    project: str,
    location: str,
    db_path: str,
    model_path: str,
    gt_path: str,
) -> NamedTuple('parameters', pipeline_root=str, db_file_path=str, db_table_name=str, model_file_path=str, groundtruth_file_path=str, binary_image_bucket_name=str, eval_container_uri=str):
    import sys
    from google.cloud import storage, secretmanager
    from googleapiclient import discovery
    from google.auth import default
    import google_crc32c

    def get_project_num(project_id: str) -> str:
        credentials, _ = default()
        service = discovery.build('cloudresourcemanager', 'v1', credentials=credentials)
        request = service.projects().get(projectId=project_id)
        response = request.execute()
        return response['projectNumber']

    def access_secret_version(project_id: str, secret_id: str, version_id: str):
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        try:
            response = client.access_secret_version(request={"name": name})
        except Exception as e:
            print(e)
            return None
        crc32c = google_crc32c.Checksum()
        crc32c.update(response.payload.data)
        if response.payload.data_crc32c != int(crc32c.hexdigest(), 16):
            print("Data corruption detected.")
            return None
        payload = response.payload.data.decode("utf-8")
        return payload

    project_num = get_project_num(project)
    pipeline_root = access_secret_version(project_num, "gcs-vertex-pipeline-root", "latest")
    docker_uri = access_secret_version(project_num, "ar-docker", "latest")
    eval_container_uri = access_secret_version(project_num, "container-evaluate-uri", "latest")
    table_name = access_secret_version(project_num, "image-info-table-name", "latest")

    if db_path == "prod":
        db_bucket = access_secret_version(project_num, "gcs-backup", "latest")
        db_file_name = access_secret_version(project_num, "image-info-db-filename", "latest")
        db_path = f"gs://{db_bucket}/{db_file_name}"
    else:
        if not db_path.startswith("gs://"):
            print(f"Invalid GCS URI({db_path}). It should start with 'gs://'")
            sys.exit(1)

    if model_path == "prod":
        model_bucket = access_secret_version(project_num, "gcs-backup", "latest")
        model_file_name = access_secret_version(project_num, "model-db-filename", "latest")
        model_path = f"gs://{model_bucket}/{model_file_name}"
    elif model_path == "pretrained" or model_path == "pass":
        model_path = model_path
    else:
        if not model_path.startswith("gs://"):
            print(f"Invalid GCS URI({model_path}). It should start with 'gs://'")
            sys.exit(1)

    if gt_path == "prod":
        gt_bucket = access_secret_version(project_num, "gcs-backup", "latest")
        gt_file_name = access_secret_version(project_num, "gt-filename", "latest")
        gt_path = f"gs://{gt_bucket}/{gt_file_name}"
    else:
        if not gt_path.startswith("gs://"):
            print(f"Invalid GCS URI({gt_path}). It should start with 'gs://'")
            sys.exit(1)

    binary_image_bucket = access_secret_version(project_num, "gcs-storage-binary-image", "latest")

    outputs = NamedTuple('parameters', [
        ('pipeline_root', str),
        ('db_file_path', str),
        ('db_table_name', str),
        ('model_file_path', str),
        ('groundtruth_file_path', str),
        ('binary_image_bucket_name', str),
        ('eval_container_uri', str),
    ])

    print(pipeline_root, db_path, table_name, model_path, gt_path, binary_image_bucket, f'{location}-docker.pkg.dev/{project}/{docker_uri}/{eval_container_uri}')

    return outputs(
        pipeline_root,
        db_path,
        table_name,
        model_path,
        gt_path,
        binary_image_bucket,
        f'{location}-docker.pkg.dev/{project}/{docker_uri}/{eval_container_uri}',
    )
