"""
Rutas para Auto-Scaling
========================

Endpoints para gestión de auto-scaling.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.auto_scaling import get_auto_scaler, AutoScaler

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/scaling",
    tags=["Auto-Scaling"]
)


class RecordMetricsRequest(BaseModel):
    """Request para registrar métricas"""
    cpu_usage: float = Field(..., ge=0, le=100, description="Uso de CPU (%)")
    memory_usage: float = Field(..., ge=0, le=100, description="Uso de memoria (%)")
    request_rate: float = Field(..., ge=0, description="Tasa de requests por segundo")
    queue_size: int = Field(..., ge=0, description="Tamaño de cola")


@router.post("/metrics")
async def record_metrics(
    request: RecordMetricsRequest,
    scaler: AutoScaler = Depends(get_auto_scaler)
):
    """Registrar métricas para escalado"""
    try:
        scaler.record_metrics(
            request.cpu_usage,
            request.memory_usage,
            request.request_rate,
            request.queue_size
        )
        
        return {"status": "recorded"}
    except Exception as e:
        logger.error(f"Error registrando métricas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendation")
async def get_scaling_recommendation(
    scaler: AutoScaler = Depends(get_auto_scaler)
):
    """Obtener recomendación de escalado"""
    recommendation = scaler.get_scaling_recommendation()
    return recommendation


@router.post("/scale-up")
async def scale_up(
    increment: int = 1,
    scaler: AutoScaler = Depends(get_auto_scaler)
):
    """Escalar hacia arriba"""
    success = scaler.scale_up(increment)
    
    if not success:
        raise HTTPException(status_code=400, detail="No se puede escalar más")
    
    return {
        "status": "scaled_up",
        "current_workers": scaler.current_workers
    }


@router.post("/scale-down")
async def scale_down(
    decrement: int = 1,
    scaler: AutoScaler = Depends(get_auto_scaler)
):
    """Escalar hacia abajo"""
    success = scaler.scale_down(decrement)
    
    if not success:
        raise HTTPException(status_code=400, detail="No se puede escalar menos")
    
    return {
        "status": "scaled_down",
        "current_workers": scaler.current_workers
    }


@router.get("/history")
async def get_scaling_history(
    limit: int = 100,
    scaler: AutoScaler = Depends(get_auto_scaler)
):
    """Obtener historial de escalado"""
    history = scaler.get_scaling_history(limit)
    return {"history": history}


@router.get("/status")
async def get_scaling_status(
    scaler: AutoScaler = Depends(get_auto_scaler)
):
    """Obtener estado actual de escalado"""
    return {
        "current_workers": scaler.current_workers,
        "min_workers": scaler.min_workers,
        "max_workers": scaler.max_workers,
        "scale_up_threshold": scaler.scale_up_threshold,
        "scale_down_threshold": scaler.scale_down_threshold
    }
















