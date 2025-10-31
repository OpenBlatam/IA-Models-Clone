from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import argparse
import os
import logging
from ml_data_pipeline import consume_and_store
from typing import Any, List, Dict, Optional
import asyncio
"""
ML Data Automation Script
- Automatiza el ciclo Kafka → MinIO/Ceph/HDFS → Dataset ML/LLM
- Configurable por CLI o variables de entorno
- Listo para usarse como servicio o cronjob
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    
    """main function."""
parser = argparse.ArgumentParser(description="Automatiza el pipeline de datos ML/LLM (Kafka → MinIO/Ceph/HDFS)")
    parser.add_argument('--backend', choices=['s3', 'hdfs'], required=True, help='Backend de almacenamiento (s3, hdfs)')
    parser.add_argument('--topic', type=str, required=True, help='Kafka topic a monitorear')
    parser.add_argument('--batch_size', type=int, default=100, help='Tamaño de lote para guardar')
    parser.add_argument('--bootstrap_servers', type=str, default=os.getenv('KAFKA_BOOTSTRAP', 'localhost:9092'))
    # S3/MinIO/Ceph
    parser.add_argument('--s3_bucket', type=str, default=os.getenv('S3_BUCKET'))
    parser.add_argument('--s3_prefix', type=str, default=os.getenv('S3_PREFIX', 'input_prompts'))
    parser.add_argument('--s3_endpoint', type=str, default=os.getenv('S3_ENDPOINT', 'http://localhost:9000'))
    parser.add_argument('--s3_access_key', type=str, default=os.getenv('S3_ACCESS_KEY', 'minioadmin'))
    parser.add_argument('--s3_secret_key', type=str, default=os.getenv('S3_SECRET_KEY', 'minioadmin'))
    # HDFS
    parser.add_argument('--hdfs_url', type=str, default=os.getenv('HDFS_URL', 'http://namenode:9870'))
    parser.add_argument('--hdfs_path', type=str, default=os.getenv('HDFS_PATH', '/user/ml/input_prompts.jsonl'))
    args = parser.parse_args()

    logging.info(f"Iniciando pipeline: backend={args.backend}, topic={args.topic}, batch_size={args.batch_size}")
    try:
        consume_and_store(
            topic=args.topic,
            storage_backend=args.backend,
            batch_size=args.batch_size,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            s3_endpoint=args.s3_endpoint,
            s3_access_key=args.s3_access_key,
            s3_secret_key=args.s3_secret_key,
            hdfs_url=args.hdfs_url,
            hdfs_path=args.hdfs_path,
            bootstrap_servers=[args.bootstrap_servers]
        )
    except Exception as e:
        logging.error(f"Error en el pipeline: {e}")
        raise

if __name__ == "__main__":
    main()

"""
# Ejemplo de uso como cronjob o servicio:
# python ml_data_automation.py --backend s3 --topic ml_training_examples --s3_bucket ml-datasets --s3_prefix input_prompts
# python ml_data_automation.py --backend hdfs --topic ml_training_examples --hdfs_url http://namenode:9870 --hdfs_path /user/ml/input_prompts.jsonl
""" 