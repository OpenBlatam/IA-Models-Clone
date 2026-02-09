"""
Fine-Tuning endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.fine_tuning_service import FineTuningService, FineTuningMethod, TrainingConfig

router = APIRouter()
fine_tuning_service = FineTuningService()


@router.post("/create-job")
async def create_training_job(
    model_name: str,
    method: str,
    dataset_path: str,
    output_dir: str,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    num_epochs: int = 3
) -> Dict[str, Any]:
    """Crear job de entrenamiento"""
    try:
        method_enum = FineTuningMethod(method)
        config = TrainingConfig(
            model_name=model_name,
            method=method_enum,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )
        
        job = fine_tuning_service.create_training_job(config, dataset_path, output_dir)
        
        return {
            "id": job.id,
            "model_name": job.config.model_name,
            "method": job.config.method.value,
            "status": job.status.value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start/{job_id}")
async def start_training(job_id: str) -> Dict[str, Any]:
    """Iniciar entrenamiento"""
    try:
        job = await fine_tuning_service.start_training(job_id)
        return {
            "id": job.id,
            "status": job.status.value,
            "started_at": job.started_at.isoformat() if job.started_at else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}")
async def get_training_status(job_id: str) -> Dict[str, Any]:
    """Obtener estado de entrenamiento"""
    try:
        status = fine_tuning_service.get_training_status(job_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prepare-lora")
async def prepare_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Preparar configuración LoRA"""
    try:
        config = fine_tuning_service.prepare_lora_config(r, lora_alpha, target_modules)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




