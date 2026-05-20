"""
Advanced Training endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.advanced_training import (
    AdvancedTrainingService,
    TrainingConfig,
    TrainingStrategy
)

router = APIRouter()
training_service = AdvancedTrainingService()


@router.post("/create-job")
async def create_training_job(
    job_id: str,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    use_mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
    strategy: str = "single_gpu"
) -> Dict[str, Any]:
    """Crear job de entrenamiento"""
    try:
        # Note: In real implementation, model and data loaders would be passed
        # For now, we create a placeholder config
        strategy_enum = TrainingStrategy(strategy)
        
        # This is a placeholder - in production, you'd pass actual model and loaders
        config = TrainingConfig(
            model=None,  # Would be actual model
            train_loader=None,  # Would be actual DataLoader
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            use_mixed_precision=use_mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            strategy=strategy_enum,
        )
        
        job = training_service.create_training_job(job_id, config)
        
        return {
            "job_id": job_id,
            "status": "created",
            "config": {
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "strategy": strategy,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/{job_id}")
async def train_model(job_id: str) -> Dict[str, Any]:
    """Entrenar modelo"""
    try:
        result = await training_service.train(job_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{job_id}")
async def get_training_metrics(job_id: str) -> Dict[str, Any]:
    """Obtener métricas de entrenamiento"""
    try:
        job = training_service.training_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "job_id": job_id,
            "metrics": [
                {
                    "epoch": m.epoch,
                    "train_loss": m.train_loss,
                    "val_loss": m.val_loss,
                    "train_accuracy": m.train_accuracy,
                    "val_accuracy": m.val_accuracy,
                    "learning_rate": m.learning_rate,
                }
                for m in job.get("metrics", [])
            ],
            "best_val_loss": job.get("best_val_loss"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




