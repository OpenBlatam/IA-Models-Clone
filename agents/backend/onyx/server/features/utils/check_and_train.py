from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import boto3
import logging
import subprocess
from datetime import datetime
from typing import Any, List, Dict, Optional
import asyncio
"""
Script de automatización para entrenamiento continuo ML/LLM
- Busca el archivo más reciente en MinIO (S3 compatible)
- Si hay datos nuevos, ejecuta train_from_minio.py
- Guarda el timestamp del último entrenamiento exitoso
- Listo para cronjob o watcher
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Configuración
BUCKET = os.getenv('S3_BUCKET', 'ml-datasets')
PREFIX = os.getenv('S3_PREFIX', 'input_prompts')
ENDPOINT = os.getenv('S3_ENDPOINT', 'http://localhost:9000')
ACCESS_KEY = os.getenv('S3_ACCESS_KEY', 'minioadmin')
SECRET_KEY = os.getenv('S3_SECRET_KEY', 'minioadmin')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt2')
EPOCHS = int(os.getenv('EPOCHS', '1'))
LAST_TRAINED_FILE = 'last_trained.txt'

# 1. Buscar el archivo más reciente en MinIO
s3 = boto3.client(
    's3',
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='us-east-1'
)

response = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.jsonl')]
if not files:
    logging.info('No se encontraron archivos JSONL en el bucket.')
    exit(0)

# Ordenar por fecha en el nombre del archivo (asume formato YYYY-MM-DD_HHMMSS)
def extract_datetime(key) -> Any:
    try:
        base = os.path.basename(key)
        dt_str = base.split('.')[0].split('_')[-2] + '_' + base.split('.')[0].split('_')[-1]
        return datetime.strptime(dt_str, '%Y-%m-%d_%H%M%S')
    except Exception:
        return datetime.min

files.sort(key=extract_datetime, reverse=True)
latest_file = files[0]
latest_time = extract_datetime(latest_file)

# 2. Leer el timestamp del último entrenamiento
if os.path.exists(LAST_TRAINED_FILE):
    with open(LAST_TRAINED_FILE) as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        last_trained_str = f.read().strip()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        last_trained = datetime.strptime(last_trained_str, '%Y-%m-%d_%H%M%S')
else:
    last_trained = datetime.min

# 3. Si hay un archivo más nuevo, entrenar
if latest_time > last_trained:
    logging.info(f'Nuevo archivo detectado: {latest_file}. Iniciando entrenamiento...')
    cmd = [
        'python', 'train_from_minio.py',
        '--backend', 's3',
        '--dataset_key', latest_file,
        '--s3_bucket', BUCKET,
        '--s3_endpoint', ENDPOINT,
        '--s3_access_key', ACCESS_KEY,
        '--s3_secret_key', SECRET_KEY,
        '--model_name', MODEL_NAME,
        '--epochs', str(EPOCHS)
    ]
    try:
        subprocess.run(cmd, check=True)
        with open(LAST_TRAINED_FILE, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(latest_time.strftime('%Y-%m-%d_%H%M%S'))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        logging.info('Entrenamiento exitoso y timestamp actualizado.')
    except subprocess.CalledProcessError as e:
        logging.error(f'Error en el entrenamiento: {e}')
else:
    logging.info('No hay archivos nuevos para entrenar.')

"""
# Ejemplo de uso como cronjob:
# */10 * * * * cd /ruta/a/tu/proyecto && python agents/backend/onyx/server/features/utils/check_and_train.py
""" 