"""
Rutas para Memory Optimization
================================

Endpoints para optimización de memoria.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.memory_optimization import (
    get_memory_optimization,
    MemoryOptimization,
    MemoryOptimizationMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/memory-optimization",
    tags=["Memory Optimization"]
)


@router.post("/models/{model_id}/optimize")
async def optimize_memory(
    model_id: str,
    method: str = Field("gradient_checkpointing", description="Método"),
    system: MemoryOptimization = Depends(get_memory_optimization)
):
    """Optimizar memoria de modelo"""
    try:
        opt_method = MemoryOptimizationMethod(method)
        profile = system.optimize_memory(model_id, opt_method)
        
        return {
            "profile_id": profile.profile_id,
            "model_id": profile.model_id,
            "original_memory_mb": profile.original_memory_mb,
            "optimized_memory_mb": profile.optimized_memory_mb,
            "reduction_percent": profile.reduction_percent,
            "method": profile.method.value
        }
    except Exception as e:
        logger.error(f"Error optimizando memoria: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/analyze")
async def analyze_memory(
    model_id: str,
    system: MemoryOptimization = Depends(get_memory_optimization)
):
    """Analizar uso de memoria"""
    try:
        analysis = system.analyze_memory_usage(model_id)
        
        return analysis
    except Exception as e:
        logger.error(f"Error analizando memoria: {e}")
        raise HTTPException(status_code=500, detail=str(e))


