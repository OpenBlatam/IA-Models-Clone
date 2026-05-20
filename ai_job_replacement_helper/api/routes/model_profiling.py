"""
Model Profiling endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_profiling import ModelProfilingService

router = APIRouter()
profiling_service = ModelProfilingService()


@router.post("/profile")
async def profile_model(
    input_shape: List[int],
    num_runs: int = 10
) -> Dict[str, Any]:
    """Hacer profiling de modelo"""
    try:
        # In production, this would receive actual model
        # For now, return structure
        return {
            "input_shape": input_shape,
            "num_runs": num_runs,
            "status": "ready",
            "note": "Model profiling requires actual model instance",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




