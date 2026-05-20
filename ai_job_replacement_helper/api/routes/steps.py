"""
Steps guide endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.steps_guide import StepsGuideService
from models.schemas import StepStartRequest, StepCompleteRequest

router = APIRouter()
steps_service = StepsGuideService()


@router.get("/roadmap/{user_id}")
async def get_roadmap(user_id: str) -> Dict[str, Any]:
    """Obtener roadmap completo del usuario"""
    try:
        roadmap = steps_service.get_user_roadmap(user_id)
        progress = steps_service.get_step_progress(user_id)
        return {
            "roadmap": [steps_service._step_to_dict(step) for step in roadmap],
            "progress": progress,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{user_id}")
async def get_progress(user_id: str) -> Dict[str, Any]:
    """Obtener progreso de pasos del usuario"""
    try:
        progress = steps_service.get_step_progress(user_id)
        return progress
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start/{user_id}")
async def start_step(
    user_id: str,
    request: StepStartRequest
) -> Dict[str, Any]:
    """Iniciar un paso"""
    try:
        result = steps_service.start_step(user_id, request.step_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complete/{user_id}")
async def complete_step(
    user_id: str,
    request: StepCompleteRequest
) -> Dict[str, Any]:
    """Completar un paso"""
    try:
        result = steps_service.complete_step(
            user_id,
            request.step_id,
            request.notes
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




