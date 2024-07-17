# kfp_VertexAI
Kubeflow pipeline code in google cloud platform

# reference
https://towardsdatascience.com/distributed-hyperparameter-tuning-in-vertex-ai-pipeline-2f3278a1eb64  


## requirement
kfp==2.7.0

## Getting Start

1. docker image 생성
```bash
cd kfp_VertexAI
docker build -t asia-northeast3-docker.pkg.dev/project이름/atifact_registry이름/docker_image이름 .
```

2. docker image push
```bash
docker push asia-northeast3-docker.pkg.dev/project이름/atifact_registry이름/docker_image이름
```

3. image기반의 cloud run 배포
```bash
gcloud run jobs deploy cloud_run이름 --image asia-northeast3-docker.pkg.dev/project이름/atifact_registry이름/docker_image이름:latest \
--max-retries 1 --region asia-northeast3 --project project이름 --vpc-connector private_service용_vpc이름 --vpc-egress all-traffic --cpu 2(낮게) --memory 1Gi(낮게) --task-timeout 24h
```