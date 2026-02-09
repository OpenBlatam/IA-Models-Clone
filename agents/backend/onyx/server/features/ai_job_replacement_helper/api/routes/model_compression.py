"""
Model Compression endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_compression import (
    ModelCompressionService,
    CompressionConfig
)

router = APIRouter()
compression_service = ModelCompressionService()


@router.post("/compress")
async def compress_model(
    model_id: str,
    method: str = "quantization",
    quantization_type: str = "dynamic",
    pruning_ratio: float = 0.3
) -> Dict[str, Any]:
    """Comprimir modelo"""
    try:
        config = CompressionConfig(
            method=method,
            quantization_type=quantization_type,
            pruning_ratio=pruning_ratio,
        )
        
        # In production, you would pass the actual model
        # result = compression_service.compress_model(model_id, model, config)
        
        return {
            "model_id": model_id,
            "method": method,
            "status": "ready",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




