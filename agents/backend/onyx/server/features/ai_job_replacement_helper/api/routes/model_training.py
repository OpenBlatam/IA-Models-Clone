"""
Model Training endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_training import ModelTrainingService, TrainingConfig

router = APIRouter()
training_service = ModelTrainingService()


@router.post("/create-job")
async def create_training_job(
    model_name: str,
    dataset_path: str,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    epochs: int = 10
) -> Dict[str, Any]:
    """Crear job de entrenamiento"""
    try:
        config = TrainingConfig(
            model_name=model_name,
            dataset_path=dataset_path,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
        )
        job = training_service.create_training_job(config)
        return {
            "job_id": job.id,
            "model_name": job.config.model_name,
            "status": job.status.value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/{job_id}")
async def train_model(job_id: str) -> Dict[str, Any]:
    """Entrenar modelo"""
    try:
        job = await training_service.train_model(job_id)
        return {
            "job_id": job.id,
            "status": job.status.value,
            "epochs_completed": len(job.metrics),
            "best_model_path": job.best_model_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{job_id}")
async def get_training_progress(job_id: str) -> Dict[str, Any]:
    """Obtener progreso de entrenamiento"""
    try:
        progress = training_service.get_training_progress(job_id)
        return progress
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




