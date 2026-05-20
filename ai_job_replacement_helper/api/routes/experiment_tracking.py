"""
Experiment Tracking endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.experiment_tracking import (
    ExperimentTrackingService,
    ExperimentConfig,
    TrackingBackend
)

router = APIRouter()
tracking_service = ExperimentTrackingService()


@router.post("/start")
async def start_experiment(
    experiment_id: str,
    name: str,
    project: str = "default-project",
    backend: str = "wandb",
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Iniciar nuevo experimento"""
    try:
        backend_enum = TrackingBackend(backend)
        config = ExperimentConfig(
            name=name,
            project=project,
            tags=tags or [],
            backend=backend_enum
        )
        
        experiment = tracking_service.start_experiment(experiment_id, config)
        
        return {
            "experiment_id": experiment_id,
            "name": name,
            "project": project,
            "started_at": experiment["started_at"].isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/log-metric")
async def log_metric(
    experiment_id: str,
    name: str,
    value: float,
    step: Optional[int] = None
) -> Dict[str, Any]:
    """Registrar métrica"""
    try:
        tracking_service.log_metric(experiment_id, name, value, step)
        return {
            "experiment_id": experiment_id,
            "metric": name,
            "value": value,
            "logged": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/finish")
async def finish_experiment(experiment_id: str) -> Dict[str, Any]:
    """Finalizar experimento"""
    try:
        tracking_service.finish_experiment(experiment_id)
        return {
            "experiment_id": experiment_id,
            "finished": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
