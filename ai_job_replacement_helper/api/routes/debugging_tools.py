"""
Debugging Tools endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.debugging_tools import DebuggingTools, DebuggingConfig

router = APIRouter()


@router.post("/check-health")
async def check_model_health(
    input_shape: List[int],
    device: Optional[str] = None
) -> Dict[str, Any]:
    """Verificar salud del modelo"""
    try:
        # In production, this would receive actual model
        return {
            "input_shape": input_shape,
            "device": device,
            "status": "ready",
            "note": "Model health check requires actual model instance",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activation-stats")
async def get_activation_stats() -> Dict[str, Any]:
    """Obtener estadísticas de activaciones"""
    try:
        # In production, this would get from active debugging session
        return {
            "stats": {},
            "note": "Activation stats require active debugging session",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




