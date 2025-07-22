# Vertex AI Kubeflow Pipeline Samples
이 저장소는 **Google Cloud Vertex AI**에서 **Kubeflow Pipelines**를 사용하는 MLOps pipeline를 직접 실행하거나 Pipeline template을 등록하는 코드입니다.   
폴더 계층 구조는 Mono Repository 형태로 구현되어 있습니다.

## Features

- Kubeflow Pipelines SDK를 사용하여 pipeline과 components를 모듈화
- 학습 실험 모델 관리 등을 위한 Vertex AI로 통합
- Custom container 기반의 components를 사용 for flexibility and reproducibility
- pipeline 실행을 위한 parameters의 Configuration 컴포넌트 구현
- 원활한 CI/CD를 위해서 일부 컴포넌트의 Container 구성
- Cloud Build or GitHub Actions를 사용한 CI/CD 호환성 테스트 필요

## Getting Started

- 설정하기
    ```
    ENV PYTHONPATH="/my_repository_path"
    ```

- 파이썬 라이브러리 설치하기
    ```bash
    cd /my_repository_path
    pip3 install -r pipeline/mlops_pipeline/requirements.txt
    ```

- 실행하기(예시)

    `--run_type`을 통해서 Artifact Registry로 upload하는지, trriger를 통해 파이프라인 실행 Job을 제출할 것인지를 정의.   
    `--exec_task`를 통해서 어떤 파이프라인을 대상으로 실행시킬 것인지를 정의.

    ```bash
    cd /my_repository_path
    ptyhon pipeline/mlops_pipeline/main.py --run_type upload --exec_task e2e
    ```


# References
- https://github.com/GoogleCloudPlatform/vertex-pipelines-end-to-end-samples/
- https://github.com/teamdatatonic/vertex-pipelines-end-to-end-samples
- https://towardsdatascience.com/distributed-hyperparameter-tuning-in-vertex-ai-pipeline-2f3278a1eb64
