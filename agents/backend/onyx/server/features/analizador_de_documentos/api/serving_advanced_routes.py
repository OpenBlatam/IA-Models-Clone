"""
Rutas para Model Serving Advanced
===================================

Endpoints para serving avanzado.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.model_serving_advanced import (
    get_model_serving_advanced,
    ModelServingAdvanced,
    ServingMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/model-serving-advanced",
    tags=["Model Serving Advanced"]
)


@router.post("/models/{model_id}/endpoints")
async def create_endpoint(
    model_id: str,
    serving_method: str = Field("rest_api", description="Método"),
    system: ModelServingAdvanced = Depends(get_model_serving_advanced)
):
    """Crear endpoint de serving"""
    try:
        method = ServingMethod(serving_method)
        endpoint = system.create_endpoint(model_id, method)
        
        return {
            "endpoint_id": endpoint.endpoint_id,
            "model_id": endpoint.model_id,
            "serving_method": endpoint.serving_method.value,
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
    system: ModelServingAdvanced = Depends(get_model_serving_advanced)
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
async def get_endpoint_metrics(
    endpoint_id: str,
    system: ModelServingAdvanced = Depends(get_model_serving_advanced)
):
    """Obtener métricas de endpoint"""
    try:
        metrics = system.get_endpoint_metrics(endpoint_id)
        
        return {"endpoint_id": endpoint_id, "metrics": metrics}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


