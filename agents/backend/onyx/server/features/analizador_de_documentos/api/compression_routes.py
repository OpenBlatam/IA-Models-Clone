"""
Rutas para Model Compression
==============================

Endpoints para compresión de modelos.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.model_compression import (
    get_model_compression,
    ModelCompression,
    CompressionMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/model-compression",
    tags=["Model Compression"]
)


@router.post("/compress/{model_id}")
async def compress_model(
    model_id: str,
    method: str = Field("quantization", description="Método"),
    target_ratio: float = Field(0.5, description="Ratio objetivo"),
    system: ModelCompression = Depends(get_model_compression)
):
    """Comprimir modelo"""
    try:
        comp_method = CompressionMethod(method)
        result = system.compress_model(model_id, comp_method, target_ratio)
        
        return {
            "compression_id": result.compression_id,
            "original_size_mb": result.original_size_mb,
            "compressed_size_mb": result.compressed_size_mb,
            "compression_ratio": result.compression_ratio,
            "method": result.method.value,
            "accuracy_drop": result.accuracy_drop,
            "speedup": result.speedup
        }
    except Exception as e:
        logger.error(f"Error comprimiendo modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quantize/{model_id}")
async def quantize_model(
    model_id: str,
    bits: int = Field(8, description="Bits de cuantización"),
    system: ModelCompression = Depends(get_model_compression)
):
    """Cuantizar modelo"""
    try:
        result = system.quantize_model(model_id, bits)
        
        return {
            "compression_id": result.compression_id,
            "original_size_mb": result.original_size_mb,
            "compressed_size_mb": result.compressed_size_mb,
            "bits": bits,
            "accuracy_drop": result.accuracy_drop,
            "speedup": result.speedup
        }
    except Exception as e:
        logger.error(f"Error cuantizando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prune/{model_id}")
async def prune_model(
    model_id: str,
    sparsity: float = Field(0.5, description="Esparcidad"),
    system: ModelCompression = Depends(get_model_compression)
):
    """Podar modelo"""
    try:
        result = system.prune_model(model_id, sparsity)
        
        return {
            "compression_id": result.compression_id,
            "original_size_mb": result.original_size_mb,
            "compressed_size_mb": result.compressed_size_mb,
            "sparsity": sparsity,
            "accuracy_drop": result.accuracy_drop,
            "speedup": result.speedup
        }
    except Exception as e:
        logger.error(f"Error podando modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{compression_id}")
async def get_compression_stats(
    compression_id: str,
    system: ModelCompression = Depends(get_model_compression)
):
    """Obtener estadísticas de compresión"""
    try:
        stats = system.get_compression_stats(compression_id)
        
        return stats
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


