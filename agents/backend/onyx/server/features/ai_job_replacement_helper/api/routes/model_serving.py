"""
Model Serving endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_serving import (
    ModelServingService,
    InferenceConfig
)

router = APIRouter()
serving_service = ModelServingService()


@router.post("/load-model")
async def load_model_for_serving(
    model_id: str,
    use_torchscript: bool = True,
    use_quantization: bool = False,
    batch_size: int = 1
) -> Dict[str, Any]:
    """Cargar modelo para serving"""
    try:
        # In production, you would pass the actual model
        # For now, this is a placeholder
        config = InferenceConfig(
            model_id=model_id,
            use_torchscript=use_torchscript,
            use_quantization=use_quantization,
            batch_size=batch_size,
        )
        
        # This would load the actual model
        # success = serving_service.load_model(model, config)
        
        return {
            "model_id": model_id,
            "loaded": True,
            "use_torchscript": use_torchscript,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quantize/{model_id}")
async def quantize_model(
    model_id: str,
    quantization_type: str = "dynamic"
) -> Dict[str, Any]:
    """Cuantizar modelo"""
    try:
        success = serving_service.quantize_model(model_id, quantization_type)
        return {
            "model_id": model_id,
            "quantized": success,
            "type": quantization_type,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{model_id}")
async def get_model_stats(model_id: str) -> Dict[str, Any]:
    """Obtener estadísticas del modelo"""
    try:
        stats = serving_service.get_model_stats(model_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




