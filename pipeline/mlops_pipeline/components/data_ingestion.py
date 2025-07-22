"""
데이터 인입(injection)에 관련된 Custom Kubeflow Component를 정의합니다.
GCS에 저장되어있는 학습 파일들을 다운로드합니다.
"""

from get_env import BASE_IMAGE_URI
from kfp.dsl import component, Dataset, Model, Output, OutputPath

@component(
    base_image=BASE_IMAGE_URI,
)
def download_images(
    gcs_db_path: str,
    table_name: str,
    bucket_binary_image: str,
    log_msg_prefix: str,
    log_fail_date: str,
    db_dir: OutputPath("Any"),
    all_img_dir: Output[Dataset],
):
    import os
    import sys
    import gc
    import time
    import cv2
    import pandas as pd
    import numpy as np
    import sqlite3
    import logging
    from pathlib import Path
    from typing import Optional
    from google.cloud.storage import Client, transfer_manager

    download_fail_log = {
        "event_type": "mlops_download_failed_file",
        "failed_date": log_fail_date,
        "data_src": "",
        "file": "",
        "target_bucket": "",
        "target_blob": "",
        "status": "fail"
    }

    def load_db(db_path: str, tbl_name: str="db_table", filter_a: Optional[str]=None, filter_b: Optional[str]=None):
        query = f"SELECT * FROM {tbl_name}"
        conditions = []
        if filter_a is not None:
            conditions.append(f"column_a like '%{filter_a}%'")
        if filter_b is not None:
            conditions.append(f"column_b = '{filter_b}'")
        if conditions:
            query += " where " + " AND ".join(conditions)
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql(query, conn)
        del conn
        gc.collect()
        return df

    def download_bucket_with_transfer_manager(bucket_name: str, blob_name_prefix: str, target_list: list=[], destination_directory: str="", workers: int=8, fail_logdict: dict=download_fail_log):
        storage_client = Client()
        bucket = storage_client.bucket(bucket_name)
        blob_name_prefix = blob_name_prefix if blob_name_prefix[-1] == "/" else blob_name_prefix + "/"

        if target_list == []:
            blob_names = [blob.name.replace(blob_name_prefix, '') for blob in bucket.list_blobs(prefix=blob_name_prefix) if blob.name != blob_name_prefix]
        else:
            print("지정된 다운로드 파일:", target_list)
            blob_names = target_list

        results = transfer_manager.download_many_to_path(
            bucket, blob_names, destination_directory=destination_directory, blob_name_prefix=blob_name_prefix, max_workers=workers)

        retry_blobs = []
        for name, result in zip(blob_names, results):
            if isinstance(result, Exception):
                retry_blobs.append(name)

        dest = set(os.listdir(destination_directory))
        num_downloaded = len(dest & set(blob_names))
        num_failed = len(retry_blobs)
        print(f'1차 다운로드 결과: {num_downloaded} files downloaded. (fail_download:{num_failed})')

        if len(retry_blobs) != 0:
            print(f'retry 개수 {len(retry_blobs)}')
            retry_results = transfer_manager.download_many_to_path(
                bucket, retry_blobs, destination_directory=destination_directory, blob_name_prefix=blob_name_prefix, max_workers=workers)
            for name, result in zip(retry_blobs, retry_results):
                if isinstance(result, Exception):
                    download_fail_log["file"] = name
                    logging.warning(download_fail_log)
                    print(f'Failed to download {{}} due to exception: {{}}'.format(name, result))
                    os.remove(f'{destination_directory}/{name}')
                    time.sleep(0.5)
                    num_failed -= 1
        print(f'최종 다운로드 결과: {num_downloaded} files downloaded. (fail_download:{num_failed})')

    workers = os.cpu_count()
    os.makedirs(db_dir, exist_ok=True)
    db_bucket, rest = gcs_db_path[5:].split('/', 1)
    db_file_name = Path(rest).name
    db_prefix = rest.replace(db_file_name, "")

    download_fail_log["data_src"] = "database"
    download_fail_log["target_bucket"] = db_bucket
    download_fail_log["target_blob"] = db_prefix

    download_bucket_with_transfer_manager(
        bucket_name=db_bucket,
        blob_name_prefix=db_prefix,
        target_list=[db_file_name],
        destination_directory=db_dir,
        workers=1,
        fail_logdict=download_fail_log,
    )

    df = load_db(os.path.join(db_dir, db_file_name), table_name)

    if os.path.isfile(os.path.join(db_dir, db_file_name)):
        df = df[["column_a"]]
        df = df.drop_duplicates()
    else:
        print("DB 다운로드 실패로 인한 파이프라인 종료")
        logging.error(f'{log_msg_prefix}failed. (fail_download_db)')
        sys.exit(1)

    storage_client = Client()
    bucket = storage_client.bucket(bucket_binary_image)
    print(all_img_dir.path)
    os.makedirs(all_img_dir.path, exist_ok=True)
    all_img_dir.metadata["total_num"] = len(df)

    download_fail_log["data_src"] = "bin.image"
    download_fail_log["file"] = ""
    download_fail_log["target_bucket"] = bucket_binary_image
    download_fail_log["target_blob"] = ""

    for row in df.itertuples():
        img_path = row.bin_img_path.replace(f"gs://{bucket_binary_image}/", "")
        download_fail_log["file"] = os.path.basename(img_path)
        download_fail_log["target_blob"] = os.path.dirname(img_path)
        try:
            blob = bucket.blob(img_path)
            image_bytes = blob.download_as_bytes()
            binary_image = np.frombuffer(image_bytes, dtype=np.uint8)
            binary_image = cv2.imdecode(binary_image, cv2.IMREAD_COLOR)
            cv2.imwrite(f"{all_img_dir.path}/{os.path.basename(img_path)}", binary_image)
        except Exception as e:
            logging.warning(download_fail_log)

    if len(os.listdir(all_img_dir.path)) > 0:
        total_len = len(os.listdir(all_img_dir.path))
        all_img_dir.metadata["num_download"] = total_len
        all_img_dir.metadata["num_failed"] = len(df) - total_len
        print(f"총 {total_len}개 파일 저장완료")
    else:
        all_img_dir.metadata["num_download"] = 0
        print("이미지 파일들 다운로드 전부 실패로 인한 파이프라인 종료")
        logging.error(f'{log_msg_prefix}failed. (fail_download_all.images)')
        sys.exit(1)
