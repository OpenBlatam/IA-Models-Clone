"""
Rutas para Advanced Model Compression
======================================

Endpoints para compresión avanzada de modelos.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.advanced_model_compression import (
    get_advanced_model_compression,
    AdvancedModelCompression,
    CompressionTechnique
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/advanced-model-compression",
    tags=["Advanced Model Compression"]
)


@router.post("/models/{model_id}/compress")
async def compress_model(
    model_id: str,
    technique: str = Field("quantization_int8", description="Técnica"),
    target_compression_ratio: float = Field(None, description="Ratio objetivo"),
    system: AdvancedModelCompression = Depends(get_advanced_model_compression)
):
    """Comprimir modelo"""
    try:
        comp_technique = CompressionTechnique(technique)
        result = system.compress_model(model_id, comp_technique, target_compression_ratio)
        
        return {
            "result_id": result.result_id,
            "model_id": result.model_id,
            "original_size_mb": result.original_size_mb,
            "compressed_size_mb": result.compressed_size_mb,
            "compression_ratio": result.compression_ratio,
            "accuracy_drop": result.accuracy_drop,
            "speedup": result.speedup,
            "technique": result.technique.value
        }
    except Exception as e:
        logger.error(f"Error comprimiendo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/tradeoff")
async def analyze_tradeoff(
    model_id: str,
    system: AdvancedModelCompression = Depends(get_advanced_model_compression)
):
    """Analizar trade-off compresión/precisión"""
    try:
        analysis = system.analyze_compression_tradeoff(model_id)
        
        return analysis
    except Exception as e:
        logger.error(f"Error analizando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


