"""
Script de Entrenamiento
=======================

Script para entrenar modelos de generación de manuales.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Agregar al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.training.trainer import ManualTrainer, ManualDataset
from ml.config.ml_config import get_ml_config
from database.session import get_async_session, init_db
from database.models import Manual
from sqlalchemy import select

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def prepare_training_data():
    """Preparar datos de entrenamiento desde BD."""
    init_db()
    
    async for db in get_async_session():
        try:
            # Obtener manuales públicos
            query = select(Manual).where(
                Manual.is_public == True,
                Manual.manual_content.isnot(None)
            ).limit(1000)  # Limitar para entrenamiento inicial
            
            result = await db.execute(query)
            manuals = list(result.scalars().all())
            
            problems = []
            manual_texts = []
            
            for manual in manuals:
                problems.append(manual.problem_description)
                manual_texts.append(manual.manual_content)
            
            logger.info(f"Preparados {len(problems)} ejemplos de entrenamiento")
            return problems, manual_texts
        
        except Exception as e:
            logger.error(f"Error preparando datos: {str(e)}")
            return [], []


def main():
    """Función principal de entrenamiento."""
    config = get_ml_config()
    
    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO DE MODELO DE MANUALES")
    logger.info("=" * 60)
    
    # Preparar datos
    logger.info("Preparando datos de entrenamiento...")
    problems, manuals = asyncio.run(prepare_training_data())
    
    if len(problems) < 10:
        logger.error("No hay suficientes datos para entrenar (mínimo 10 ejemplos)")
        return
    
    # Inicializar trainer
    logger.info(f"Inicializando trainer con modelo: {config.generation_model}")
    trainer = ManualTrainer(
        model_name=config.generation_model,
        use_lora=config.use_lora,
        use_wandb=config.use_wandb,
        project_name=config.wandb_project
    )
    
    # Preparar datasets
    logger.info("Preparando datasets...")
    full_dataset = trainer.prepare_dataset(problems, manuals, max_length=config.max_length)
    
    # Split train/val (80/20)
    split_idx = int(len(full_dataset) * 0.8)
    train_dataset = ManualDataset(
        full_dataset.texts[:split_idx],
        trainer.tokenizer,
        config.max_length
    )
    val_dataset = ManualDataset(
        full_dataset.texts[split_idx:],
        trainer.tokenizer,
        config.max_length
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Entrenar
    logger.info("Iniciando entrenamiento...")
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        output_dir=config.models_dir
    )
    
    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()




