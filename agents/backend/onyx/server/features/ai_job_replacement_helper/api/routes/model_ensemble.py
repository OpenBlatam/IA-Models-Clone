"""
Model Ensemble endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_ensemble import (
    ModelEnsembleService,
    EnsembleConfig
)

router = APIRouter()
ensemble_service = ModelEnsembleService()


@router.post("/create")
async def create_ensemble(
    ensemble_id: str,
    method: str = "averaging",
    weights: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Crear ensemble"""
    try:
        config = EnsembleConfig(
            method=method,
            weights=weights,
        )
        
        # In production, you would pass actual models
        # success = ensemble_service.create_ensemble(ensemble_id, models, config)
        
        return {
            "ensemble_id": ensemble_id,
            "method": method,
            "weights": weights,
            "status": "ready",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info/{ensemble_id}")
async def get_ensemble_info(ensemble_id: str) -> Dict[str, Any]:
    """Obtener información del ensemble"""
    try:
        info = ensemble_service.get_ensemble_info(ensemble_id)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




