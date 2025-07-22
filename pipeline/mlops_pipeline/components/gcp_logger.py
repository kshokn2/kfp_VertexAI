"""
로깅에 관련 Custom Kubeflow Component를 정의합니다.
"""

from get_env import BASE_IMAGE_URI
from kfp.dsl import component

@component(
    base_image=BASE_IMAGE_URI,
)
def mlops_msg_log(msg: str, severity: str):
    """
    Kubeflow의 재학습 파이프라인의 결과를 기록하는 Log Operator
    예시. '[Task #mlops-pipeline] finished. (1차 평가 결과 및 모델 배포: 승인)'
        또는 '[Task #mlops-pipeline] finished. (3차 평가 결과 및 모델 배포: 중단)' 등의 message를 기록.

    Args:
        msg(str): 로그 메시지.
        severity(str): 로그 등급.
    """

    import logging

    if severity == "DEBUG":
        logging.debug(msg)
    elif severity == "INFO":
        logging.info(msg)
    elif severity == "WARNING":
        logging.warning(msg)
    elif severity == "ERROR":
        logging.error(msg)
    elif severity == "CRITICAL":
        logging.critical(msg)
    else:
        logging.info(msg)
