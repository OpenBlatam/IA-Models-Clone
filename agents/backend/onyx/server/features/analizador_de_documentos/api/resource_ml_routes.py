"""
Rutas para ML Resource Optimization
=====================================

Endpoints para optimización de recursos de ML.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.ml_resource_optimization import (
    get_ml_resource_optimization,
    MLResourceOptimization,
    ResourceType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/ml-resource-optimization",
    tags=["ML Resource Optimization"]
)


@router.post("/models/{model_id}/optimize")
async def optimize_allocation(
    model_id: str,
    resource_type: str = Field(..., description="Tipo de recurso"),
    current_usage: float = Field(..., description="Uso actual"),
    system: MLResourceOptimization = Depends(get_ml_resource_optimization)
):
    """Optimizar asignación de recursos"""
    try:
        res_type = ResourceType(resource_type)
        allocation = system.optimize_allocation(model_id, res_type, current_usage)
        
        return {
            "allocation_id": allocation.allocation_id,
            "model_id": allocation.model_id,
            "resource_type": allocation.resource_type.value,
            "allocated_amount": allocation.allocated_amount,
            "optimal_amount": allocation.optimal_amount,
            "efficiency": allocation.efficiency
        }
    except Exception as e:
        logger.error(f"Error optimizando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/usage")
async def analyze_usage(
    model_id: str,
    system: MLResourceOptimization = Depends(get_ml_resource_optimization)
):
    """Analizar uso de recursos"""
    try:
        analysis = system.analyze_resource_usage(model_id)
        
        return analysis
    except Exception as e:
        logger.error(f"Error analizando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/recommendations")
async def get_recommendations(
    model_id: str,
    system: MLResourceOptimization = Depends(get_ml_resource_optimization)
):
    """Obtener recomendaciones de optimización"""
    try:
        recommendations = system.get_optimization_recommendations(model_id)
        
        return {"model_id": model_id, "recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error obteniendo recomendaciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


