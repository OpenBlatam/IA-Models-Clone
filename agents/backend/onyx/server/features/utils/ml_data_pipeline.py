from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import json
from datetime import datetime
from typing import List
from kafka import KafkaProducer, KafkaConsumer
import boto3
from hdfs import InsecureClient
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
ML/LLM Data Pipeline Utilities
- Kafka ingestion
- MinIO/Ceph (S3 compatible) and HDFS storage
- Ready for training/fine-tuning workflows
"""

# Kafka

# MinIO/Ceph (S3 compatible)

# HDFS

# --- Kafka Producer ---
def send_training_example_kafka(instance, topic="ml_training_examples", bootstrap_servers=None) -> Any:
    """Send a training example to a Kafka topic."""
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers or ['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    example = instance.to_training_example()
    producer.send(topic, example)
    producer.flush()

# --- Kafka Consumer & Storage ---
def consume_and_store(
    topic: str,
    storage_backend: str,
    batch_size: int = 100,
    s3_bucket: str = None,
    s3_prefix: str = None,
    s3_endpoint: str = None,
    s3_access_key: str = None,
    s3_secret_key: str = None,
    hdfs_url: str = None,
    hdfs_path: str = None,
    bootstrap_servers=None
):
    """
    Consume from Kafka and store batches in MinIO/Ceph (S3) or HDFS.
    storage_backend: 's3' or 'hdfs'
    """
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers or ['localhost:9092'],
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='ml-data-collector'
    )
    batch = []
    for message in consumer:
        batch.append(message.value)
        if len(batch) >= batch_size:
            if storage_backend == 's3':
                save_batch_to_s3(batch, s3_bucket, s3_prefix, s3_endpoint, s3_access_key, s3_secret_key)
            elif storage_backend == 'hdfs':
                save_batch_to_hdfs(batch, hdfs_url, hdfs_path)
            batch.clear()

# --- MinIO/Ceph (S3) ---
def save_batch_to_s3(batch: List[dict], bucket, prefix, endpoint_url, access_key, secret_key):
    """Save a batch of examples to MinIO/Ceph (S3 compatible) as JSONL."""
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='us-east-1'
    )
    key = f"{prefix}/{datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')}.jsonl"
    content = "\n".join(json.dumps(x, ensure_ascii=False) for x in batch)
    s3.put_object(Bucket=bucket, Key=key, Body=content.encode('utf-8'))
    print(f"Saved batch to S3: s3://{bucket}/{key}")

# --- HDFS ---
def save_batch_to_hdfs(batch: List[dict], hdfs_url, hdfs_path):
    """Save a batch of examples to HDFS as JSONL."""
    client = InsecureClient(hdfs_url)
    with client.write(hdfs_path, encoding='utf-8', append=True) as writer:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        for example in batch:
            writer.write(json.dumps(example, ensure_ascii=False) + '\n')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    print(f"Saved batch to HDFS: {hdfs_path}")

# --- Read dataset from S3 ---
def load_dataset_from_s3(bucket, key, endpoint_url, access_key, secret_key) -> Any:
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='us-east-1'
    )
    obj = s3.get_object(Bucket=bucket, Key=key)
    lines = obj['Body'].read().decode('utf-8').splitlines()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    return [json.loads(line) for line in lines]

# --- Read dataset from HDFS ---
def load_dataset_from_hdfs(hdfs_url, hdfs_path) -> Any:
    client = InsecureClient(hdfs_url)
    with client.read(hdfs_path, encoding='utf-8') as reader:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        return [json.loads(line) for line in reader]

# --- Ejemplo de uso con InputPrompt ---
# from agents.backend.onyx.server.features.input_prompt.models import InputPrompt
# prompt = InputPrompt(prompt="¿Cuál es la capital de Francia?")
# send_training_example_kafka(prompt, topic="ml_training_examples", bootstrap_servers=["localhost:9092"])

# Para consumir y guardar en MinIO:
# consume_and_store(
#     topic="ml_training_examples",
#     storage_backend='s3',
#     batch_size=100,
#     s3_bucket='ml-datasets',
#     s3_prefix='input_prompts',
#     s3_endpoint='http://localhost:9000',
#     s3_access_key='minioadmin',
#     s3_secret_key='minioadmin',
#     bootstrap_servers=["localhost:9092"]
# )

# Para consumir y guardar en HDFS:
# consume_and_store(
#     topic="ml_training_examples",
#     storage_backend='hdfs',
#     batch_size=100,
#     hdfs_url='http://namenode:9870',
#     hdfs_path='/user/ml/input_prompts.jsonl',
#     bootstrap_servers=["localhost:9092"]
# ) 