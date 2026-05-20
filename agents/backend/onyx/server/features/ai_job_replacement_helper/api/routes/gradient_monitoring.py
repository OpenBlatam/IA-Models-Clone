"""
Gradient Monitoring endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.gradient_monitoring import GradientMonitoringService

router = APIRouter()
gradient_service = GradientMonitoringService()


@router.get("/summary")
async def get_gradient_summary() -> Dict[str, Any]:
    """Obtener resumen de gradientes"""
    try:
        summary = gradient_service.get_gradient_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vanishing")
async def detect_vanishing_gradients(
    threshold: float = 1e-6
) -> Dict[str, Any]:
    """Detectar gradientes que desaparecen"""
    try:
        layers = gradient_service.detect_vanishing_gradients(threshold)
        return {
            "vanishing_layers": layers,
            "count": len(layers),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exploding")
async def detect_exploding_gradients(
    threshold: float = 100.0
) -> Dict[str, Any]:
    """Detectar gradientes que explotan"""
    try:
        layers = gradient_service.detect_exploding_gradients(threshold)
        return {
            "exploding_layers": layers,
            "count": len(layers),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




