"""
Rutas para Model Serving
==========================

Endpoints para model serving.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.model_serving import (
    get_model_serving,
    ModelServing,
    ServingStrategy
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/model-serving",
    tags=["Model Serving"]
)


@router.post("/endpoints")
async def create_endpoint(
    model_id: str = Field(..., description="ID del modelo"),
    strategy: str = Field("real_time", description="Estrategia"),
    endpoint_id: str = Field(None, description="ID del endpoint"),
    system: ModelServing = Depends(get_model_serving)
):
    """Crear endpoint de serving"""
    try:
        serving_strategy = ServingStrategy(strategy)
        endpoint = system.create_endpoint(model_id, serving_strategy, endpoint_id)
        
        return {
            "endpoint_id": endpoint.endpoint_id,
            "model_id": endpoint.model_id,
            "strategy": endpoint.strategy.value,
            "url": endpoint.url,
            "status": endpoint.status
        }
    except Exception as e:
        logger.error(f"Error creando endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/endpoints/{endpoint_id}/predict")
async def serve_prediction(
    endpoint_id: str,
    input_data: Dict[str, Any] = Field(..., description="Datos de entrada"),
    system: ModelServing = Depends(get_model_serving)
):
    """Servir predicción"""
    try:
        prediction = system.serve_prediction(endpoint_id, input_data)
        
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error sirviendo predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/endpoints/{endpoint_id}/metrics")
async def get_metrics(
    endpoint_id: str,
    system: ModelServing = Depends(get_model_serving)
):
    """Obtener métricas del endpoint"""
    try:
        metrics = system.get_endpoint_metrics(endpoint_id)
        
        return metrics
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


