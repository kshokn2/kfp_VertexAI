"""
GCP 환경에서의 Python SDK를 사용하는 공통 모듈을 정의합니다.
Google Cloud의 Docs를 참고하여 필요한 함수만 가져옵니다.
"""


def access_secret_version(project_num, secret_id, version):
    """
    google cloud의 secret 접근 함수
    참고: https://cloud.google.com/secret-manager/docs/access-secret-version?hl=ko#secretmanager-access-secret-version-python
    """

def get_project_num(project_id):
    """
    google cloud에서의 project id를 통해 project num을 구하는 함수

    from googleapiclient import discovery
    from google.auth import default
    import google_crc32c

    credentials, _ = default()
    service = discovery.build('cloudresourcemanager', 'v1', credentials=credentials)
    request = service.projects().get(projectId=project_id)
    response = request.execute()
    return response['projectNumber']
    """


def download_bucket_with_transfer_manager():
    """
    transfer를 사용해서 bucket 내의 파일들을 가져오는 함수
    https://cloud.google.com/storage/docs/samples/storage-transfer-manager-download-many?hl=ko#code-sample
    
    from google.cloud.storage import Client, transfer_manager
    """
