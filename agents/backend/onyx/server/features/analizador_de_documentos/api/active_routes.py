"""
Rutas para Active Learning
===========================

Endpoints para active learning.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.active_learning import (
    get_active_learning,
    ActiveLearning,
    QueryStrategy
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/active-learning",
    tags=["Active Learning"]
)


class QuerySamplesRequest(BaseModel):
    """Request para consultar muestras"""
    unlabeled_pool: List[Dict[str, Any]] = Field(..., description="Pool no etiquetado")
    num_samples: int = Field(10, description="Número de muestras")
    strategy: str = Field("uncertainty", description="Estrategia")


@router.post("/query")
async def query_samples(
    request: QuerySamplesRequest,
    system: ActiveLearning = Depends(get_active_learning)
):
    """Consultar muestras para etiquetar"""
    try:
        strategy = QueryStrategy(request.strategy)
        result = system.query_samples(
            request.unlabeled_pool,
            request.num_samples,
            strategy
        )
        
        return {
            "query_id": result.query_id,
            "selected_samples": result.selected_samples,
            "strategy": result.strategy.value,
            "uncertainty_scores": result.uncertainty_scores
        }
    except Exception as e:
        logger.error(f"Error consultando muestras: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-labeled")
async def add_labeled_data(
    samples: List[Dict[str, Any]] = Field(..., description="Muestras"),
    labels: List[Any] = Field(..., description="Etiquetas"),
    system: ActiveLearning = Depends(get_active_learning)
):
    """Agregar datos etiquetados"""
    try:
        system.add_labeled_data(samples, labels)
        
        return {"status": "added", "samples": len(samples)}
    except Exception as e:
        logger.error(f"Error agregando datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/efficiency")
async def get_efficiency(
    system: ActiveLearning = Depends(get_active_learning)
):
    """Obtener eficiencia de etiquetado"""
    efficiency = system.get_labeling_efficiency()
    return efficiency



