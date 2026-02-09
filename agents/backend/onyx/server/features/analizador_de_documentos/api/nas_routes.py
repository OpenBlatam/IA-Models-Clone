"""
Rutas para Neural Architecture Search
=======================================

Endpoints para NAS.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.neural_architecture_search import (
    get_nas,
    NeuralArchitectureSearch,
    NASStrategy
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/nas",
    tags=["Neural Architecture Search"]
)


class SearchArchitectureRequest(BaseModel):
    """Request para buscar arquitectura"""
    search_space: Dict[str, Any] = Field(..., description="Espacio de búsqueda")
    strategy: str = Field("evolutionary", description="Estrategia de búsqueda")
    max_iterations: int = Field(100, description="Máximo de iteraciones")


@router.post("/search")
async def search_architecture(
    request: SearchArchitectureRequest,
    nas: NeuralArchitectureSearch = Depends(get_nas)
):
    """Buscar arquitectura óptima"""
    try:
        strategy = NASStrategy(request.strategy)
        architecture = nas.search_architecture(
            request.search_space,
            strategy,
            request.max_iterations
        )
        
        return {
            "architecture_id": architecture.architecture_id,
            "layers": architecture.layers,
            "parameters": architecture.parameters,
            "accuracy": architecture.accuracy,
            "latency_ms": architecture.latency_ms
        }
    except Exception as e:
        logger.error(f"Error buscando arquitectura: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_search_history(
    nas: NeuralArchitectureSearch = Depends(get_nas)
):
    """Obtener historial de búsqueda"""
    history = nas.get_search_history()
    
    return {"history": history, "total_iterations": len(history)}



