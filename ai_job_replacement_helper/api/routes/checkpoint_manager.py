"""
Checkpoint Manager endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.checkpoint_manager import CheckpointManager

router = APIRouter()
checkpoint_manager = CheckpointManager()


@router.get("/list")
async def list_checkpoints() -> Dict[str, Any]:
    """Listar checkpoints disponibles"""
    try:
        checkpoints = checkpoint_manager.list_checkpoints()
        return {
            "checkpoints": checkpoints,
            "count": len(checkpoints),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best")
async def get_best_checkpoint() -> Dict[str, Any]:
    """Obtener mejor checkpoint"""
    try:
        best_path = checkpoint_manager.get_best_checkpoint()
        return {
            "best_checkpoint": best_path,
            "exists": best_path is not None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




