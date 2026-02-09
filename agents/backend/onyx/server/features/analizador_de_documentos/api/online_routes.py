"""
Rutas para Online Learning
===========================

Endpoints para online learning.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.online_learning import (
    get_online_learning,
    OnlineLearning,
    OnlineLearningMethod,
    StreamingSample
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/online-learning",
    tags=["Online Learning"]
)


class CreateModelRequest(BaseModel):
    """Request para crear modelo"""
    method: str = Field("sgd", description="Método")


@router.post("/models/{model_id}")
async def create_model(
    model_id: str,
    request: CreateModelRequest,
    system: OnlineLearning = Depends(get_online_learning)
):
    """Crear modelo de online learning"""
    try:
        method = OnlineLearningMethod(request.method)
        model = system.create_online_model(model_id, method)
        
        return model
    except Exception as e:
        logger.error(f"Error creando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/update")
async def update_model(
    model_id: str,
    sample_id: str = Field(..., description="ID de muestra"),
    features: Dict[str, Any] = Field(..., description="Características"),
    label: Any = Field(None, description="Etiqueta"),
    learning_rate: float = Field(0.01, description="Tasa de aprendizaje"),
    system: OnlineLearning = Depends(get_online_learning)
):
    """Actualizar modelo con nueva muestra"""
    try:
        sample = StreamingSample(
            sample_id=sample_id,
            features=features,
            label=label
        )
        
        result = system.update_model(model_id, sample, learning_rate)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error actualizando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_id}/update-batch")
async def update_batch(
    model_id: str,
    samples: List[Dict[str, Any]] = Field(..., description="Muestras"),
    learning_rate: float = Field(0.01, description="Tasa de aprendizaje"),
    system: OnlineLearning = Depends(get_online_learning)
):
    """Actualizar modelo con batch"""
    try:
        streaming_samples = [
            StreamingSample(
                sample_id=sample.get("sample_id", f"sample_{i}"),
                features=sample.get("features", {}),
                label=sample.get("label")
            )
            for i, sample in enumerate(samples)
        ]
        
        result = system.update_batch(model_id, streaming_samples, learning_rate)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error actualizando batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/status")
async def get_model_status(
    model_id: str,
    system: OnlineLearning = Depends(get_online_learning)
):
    """Obtener estado del modelo"""
    try:
        status = system.get_model_status(model_id)
        
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


