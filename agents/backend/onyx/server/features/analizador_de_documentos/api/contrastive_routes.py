"""
Rutas para Contrastive Learning
==================================

Endpoints para contrastive learning.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.contrastive_learning import (
    get_contrastive_learning,
    ContrastiveLearning,
    ContrastiveMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/contrastive-learning",
    tags=["Contrastive Learning"]
)


class TrainContrastiveRequest(BaseModel):
    """Request para entrenar contrastivo"""
    data: List[Dict[str, Any]] = Field(..., description="Datos")
    method: str = Field("simclr", description="Método")
    epochs: int = Field(10, description="Número de épocas")


@router.post("/generate-pairs")
async def generate_pairs(
    data: List[Dict[str, Any]] = Field(..., description="Datos"),
    system: ContrastiveLearning = Depends(get_contrastive_learning)
):
    """Generar pares contrastivos"""
    try:
        pairs = system.generate_pairs(data)
        
        return {
            "num_pairs": len(pairs),
            "pairs": [
                {
                    "anchor": pair.anchor,
                    "positive": pair.positive,
                    "num_negatives": len(pair.negatives)
                }
                for pair in pairs[:10]  # Limitar a 10 para respuesta
            ]
        }
    except Exception as e:
        logger.error(f"Error generando pares: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def train_contrastive(
    request: TrainContrastiveRequest,
    system: ContrastiveLearning = Depends(get_contrastive_learning)
):
    """Entrenar modelo contrastivo"""
    try:
        pairs = system.generate_pairs(request.data)
        method = ContrastiveMethod(request.method)
        result = system.train_contrastive(pairs, method, request.epochs)
        
        return result
    except Exception as e:
        logger.error(f"Error entrenando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings")
async def get_embeddings(
    data: List[Dict[str, Any]] = Field(..., description="Datos"),
    run_id: str = Field(None, description="ID del entrenamiento"),
    system: ContrastiveLearning = Depends(get_contrastive_learning)
):
    """Obtener embeddings"""
    try:
        embeddings = system.get_embeddings(data, run_id)
        
        return {
            "num_embeddings": len(embeddings),
            "embedding_dim": len(embeddings[0]) if embeddings else 0,
            "embeddings": embeddings
        }
    except Exception as e:
        logger.error(f"Error obteniendo embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))



