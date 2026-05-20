"""
Model Export endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.utils.export_utils import (
    export_to_onnx,
    export_to_torchscript,
    export_model_summary,
)

router = APIRouter()


@router.post("/onnx")
async def export_onnx_endpoint(
    model_id: str,
    input_shape: List[int],
    output_path: str
) -> Dict[str, Any]:
    """Exportar modelo a ONNX"""
    try:
        # In production, you would load and export the actual model
        # success = export_to_onnx(model, tuple(input_shape), output_path)
        
        return {
            "model_id": model_id,
            "format": "onnx",
            "output_path": output_path,
            "status": "ready",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/torchscript")
async def export_torchscript_endpoint(
    model_id: str,
    input_shape: List[int],
    output_path: str,
    method: str = "trace"
) -> Dict[str, Any]:
    """Exportar modelo a TorchScript"""
    try:
        return {
            "model_id": model_id,
            "format": "torchscript",
            "method": method,
            "output_path": output_path,
            "status": "ready",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




