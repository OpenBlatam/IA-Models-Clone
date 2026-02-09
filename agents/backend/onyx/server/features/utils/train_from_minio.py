from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import argparse
import os
import logging
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import boto3
from hdfs import InsecureClient
from typing import Any, List, Dict, Optional
import asyncio
"""
Entrenamiento/Fine-tuning automático de LLM desde MinIO/Ceph/HDFS
- Descarga el dataset generado por los modelos (JSONL)
- Tokeniza y entrena/fine-tunea un modelo HuggingFace (por defecto GPT-2)
- Guarda el modelo y tokenizer resultantes
- Configurable por CLI/env
"""

# MinIO/Ceph (S3 compatible)
# HDFS

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --- Dataset utils ---
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

def load_dataset_from_hdfs(hdfs_url, hdfs_path) -> Any:
    client = InsecureClient(hdfs_url)
    with client.read(hdfs_path, encoding='utf-8') as reader:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        return [json.loads(line) for line in reader]

class PromptDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128, input_key='input', output_key=None) -> Any:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_key = input_key
        self.output_key = output_key

    def __len__(self) -> Any:
        return len(self.samples)

    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        sample = self.samples[idx]
        if self.output_key and self.output_key in sample:
            text = sample[self.input_key] + "\n" + sample[self.output_key]
        else:
            text = sample[self.input_key]
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
        }

# --- Main training script ---
def main():
    
    """main function."""
parser = argparse.ArgumentParser(description="Fine-tuning LLM desde MinIO/Ceph/HDFS")
    parser.add_argument('--backend', choices=['s3', 'hdfs'], required=True)
    parser.add_argument('--dataset_key', type=str, required=True, help='Archivo JSONL en S3/HDFS')
    parser.add_argument('--input_key', type=str, default='input', help='Campo de entrada (prompt)')
    parser.add_argument('--output_key', type=str, default=None, help='Campo de salida (opcional, para prompt+respuesta)')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='./fine_tuned_model')
    # S3/MinIO/Ceph
    parser.add_argument('--s3_bucket', type=str, default=os.getenv('S3_BUCKET'))
    parser.add_argument('--s3_endpoint', type=str, default=os.getenv('S3_ENDPOINT', 'http://localhost:9000'))
    parser.add_argument('--s3_access_key', type=str, default=os.getenv('S3_ACCESS_KEY', 'minioadmin'))
    parser.add_argument('--s3_secret_key', type=str, default=os.getenv('S3_SECRET_KEY', 'minioadmin'))
    # HDFS
    parser.add_argument('--hdfs_url', type=str, default=os.getenv('HDFS_URL', 'http://namenode:9870'))
    args = parser.parse_args()

    logging.info(f"Cargando dataset desde {args.backend}...")
    if args.backend == 's3':
        samples = load_dataset_from_s3(
            args.s3_bucket, args.dataset_key, args.s3_endpoint, args.s3_access_key, args.s3_secret_key)
    else:
        samples = load_dataset_from_hdfs(args.hdfs_url, args.dataset_key)
    logging.info(f"Ejemplos cargados: {len(samples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = PromptDataset(samples, tokenizer, max_length=args.max_length, input_key=args.input_key, output_key=args.output_key)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        save_steps=100,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    logging.info("Iniciando entrenamiento...")
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info(f"Modelo y tokenizer guardados en {args.output_dir}")

if __name__ == "__main__":
    main()

"""
# Ejemplo de uso:
# python train_from_minio.py --backend s3 --dataset_key input_prompts/2024-06-01_120000.jsonl --s3_bucket ml-datasets --model_name gpt2 --epochs 1
# python train_from_minio.py --backend hdfs --dataset_key /user/ml/input_prompts.jsonl --hdfs_url http://namenode:9870 --model_name gpt2 --epochs 1
""" 