"""
AutoML endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.automl_service import AutoMLService, AutoMLConfig, TaskType

router = APIRouter()
automl_service = AutoMLService()


@router.post("/run")
async def run_automl(
    job_id: str,
    task_type: str = "classification",
    time_budget: int = 3600,
    max_models: int = 10
) -> Dict[str, Any]:
    """Ejecutar AutoML"""
    try:
        task_enum = TaskType(task_type)
        config = AutoMLConfig(
            task_type=task_enum,
            time_budget=time_budget,
            max_models=max_models,
        )
        
        # In production, you would pass actual data
        # result = automl_service.run_automl(job_id, X_train, y_train, config=config)
        
        return {
            "job_id": job_id,
            "task_type": task_type,
            "status": "started",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{job_id}")
async def get_automl_result(job_id: str) -> Dict[str, Any]:
    """Obtener resultado de AutoML"""
    try:
        result = automl_service.get_automl_result(job_id)
        if not result:
            return {"error": "Job not found"}
        
        return {
            "job_id": job_id,
            "best_model": result.best_model,
            "best_score": result.best_score,
            "models_tested": result.models_tested,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




