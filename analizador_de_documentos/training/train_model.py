"""
Script de Entrenamiento para Fine-Tuning
=========================================

Script para entrenar modelos con fine-tuning desde línea de comandos.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Agregar path al módulo
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fine_tuning_model import FineTuningModel, FineTuningConfig
from core.document_processor import DocumentProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_training_data(data_path: str) -> tuple:
    """
    Cargar datos de entrenamiento
    
    Formato esperado: JSON con lista de objetos {"text": "...", "label": ...}
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    
    return texts, labels


def create_sample_data(output_path: str, num_samples: int = 100):
    """Crear datos de ejemplo para testing"""
    import random
    
    sample_texts = [
        "Este es un documento sobre tecnología e innovación.",
        "La inteligencia artificial está transformando la industria.",
        "El cambio climático es uno de los desafíos más importantes.",
        "La educación es fundamental para el desarrollo social.",
        "La salud pública requiere atención prioritaria.",
        "Las empresas están adoptando nuevas tecnologías.",
        "El deporte es importante para mantener una vida saludable.",
        "La cultura y el arte enriquecen nuestras vidas.",
        "La economía global está experimentando cambios significativos.",
        "La ciencia y la investigación impulsan el progreso.",
    ]
    
    # Generar datos de ejemplo
    data = []
    for i in range(num_samples):
        text = random.choice(sample_texts)
        # Simular variaciones
        text = text.replace(".", " " + random.choice(["y", "además", "también"]) + ".")
        label = random.randint(0, 2)  # 3 clases
        data.append({"text": text, "label": label})
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Datos de ejemplo creados en {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Entrenar modelo de fine-tuning para análisis de documentos"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Ruta al archivo JSON con datos de entrenamiento"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-multilingual-cased",
        help="Nombre del modelo base"
    )
    
    parser.add_argument(
        "--num-labels",
        type=int,
        default=2,
        help="Número de clases"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio de salida para el modelo entrenado"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Número de épocas"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Crear datos de ejemplo en lugar de entrenar"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Número de muestras para datos de ejemplo"
    )
    
    args = parser.parse_args()
    
    # Crear datos de ejemplo si se solicita
    if args.create_sample:
        create_sample_data(args.data, args.sample_size)
        return
    
    # Verificar que existe el archivo de datos
    if not os.path.exists(args.data):
        logger.error(f"Archivo de datos no encontrado: {args.data}")
        logger.info("Usa --create-sample para crear datos de ejemplo")
        return
    
    # Cargar datos
    logger.info(f"Cargando datos desde {args.data}")
    texts, labels = load_training_data(args.data)
    logger.info(f"Cargados {len(texts)} ejemplos")
    
    # Configurar directorio de salida
    if args.output_dir is None:
        args.output_dir = os.path.join(
            Path(__file__).parent.parent,
            "models",
            "fine_tuned",
            f"model_{Path(args.data).stem}"
        )
    
    # Crear configuración
    config = FineTuningConfig(
        model_name=args.model_name,
        num_labels=args.num_labels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    # Crear modelo
    logger.info("Inicializando modelo...")
    model = FineTuningModel(config=config)
    
    # Preparar dataset
    logger.info("Preparando dataset...")
    train_dataset, eval_dataset = model.prepare_dataset(texts, labels)
    
    # Entrenar
    logger.info("Iniciando entrenamiento...")
    results = model.train(train_dataset, eval_dataset)
    
    # Mostrar resultados
    logger.info("=" * 50)
    logger.info("Entrenamiento completado!")
    logger.info("=" * 50)
    logger.info(f"Train Loss: {results.get('train_loss', 'N/A')}")
    logger.info(f"Train Runtime: {results.get('train_runtime', 'N/A')}")
    logger.info(f"Eval Accuracy: {results.get('eval_accuracy', 'N/A')}")
    logger.info(f"Eval F1: {results.get('eval_f1', 'N/A')}")
    logger.info(f"Eval Precision: {results.get('eval_precision', 'N/A')}")
    logger.info(f"Eval Recall: {results.get('eval_recall', 'N/A')}")
    logger.info(f"Modelo guardado en: {args.output_dir}")


if __name__ == "__main__":
    main()
















