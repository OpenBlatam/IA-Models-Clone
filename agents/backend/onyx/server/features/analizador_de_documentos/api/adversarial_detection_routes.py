"""
Rutas para Adversarial Detection
==================================

Endpoints para detección de ataques adversariales.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.adversarial_detection import (
    get_adversarial_detection,
    AdversarialDetection
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/adversarial-detection",
    tags=["Adversarial Detection"]
)


@router.post("/detect")
async def detect_adversarial(
    model_id: str = Field(..., description="ID del modelo"),
    input_data: Dict[str, Any] = Field(..., description="Datos de entrada"),
    system: AdversarialDetection = Depends(get_adversarial_detection)
):
    """Detectar ataque adversarial"""
    try:
        alert = system.detect_adversarial(input_data, model_id)
        
        if alert:
            return {
                "alert_id": alert.alert_id,
                "attack_type": alert.attack_type.value,
                "confidence": alert.confidence,
                "severity": alert.severity,
                "detected": True
            }
        else:
            return {"detected": False, "message": "No se detectó ataque adversarial"}
    except Exception as e:
        logger.error(f"Error detectando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/stats")
async def get_detection_stats(
    model_id: str,
    system: AdversarialDetection = Depends(get_adversarial_detection)
):
    """Obtener estadísticas de detección"""
    try:
        stats = system.get_detection_stats(model_id)
        
        return stats
    except Exception as e:
        logger.error(f"Error obteniendo stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


