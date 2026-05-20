"""
Learning Path endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.learning_path import LearningPathService

router = APIRouter()
learning_path_service = LearningPathService()


@router.post("/create/{user_id}")
async def create_learning_path(
    user_id: str,
    title: str,
    description: str,
    skill_focus: str,
    modules: Optional[List[Dict[str, Any]]] = None,
    estimated_duration_days: int = 30
) -> Dict[str, Any]:
    """Crear ruta de aprendizaje"""
    try:
        path = learning_path_service.create_learning_path(
            user_id, title, description, skill_focus, modules, estimated_duration_days
        )
        return {
            "id": path.id,
            "title": path.title,
            "skill_focus": path.skill_focus,
            "modules_count": len(path.modules),
            "status": path.status.value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}")
async def get_user_paths(user_id: str) -> Dict[str, Any]:
    """Obtener rutas de aprendizaje del usuario"""
    try:
        paths = learning_path_service.get_user_paths(user_id)
        return {
            "paths": [
                {
                    "id": p.id,
                    "title": p.title,
                    "skill_focus": p.skill_focus,
                    "status": p.status.value,
                    "progress_percentage": p.progress_percentage,
                }
                for p in paths
            ],
            "total": len(paths),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start/{user_id}/{path_id}")
async def start_path(user_id: str, path_id: str) -> Dict[str, Any]:
    """Iniciar ruta de aprendizaje"""
    try:
        path = learning_path_service.start_learning_path(user_id, path_id)
        return {
            "id": path.id,
            "status": path.status.value,
            "started_at": path.started_at.isoformat() if path.started_at else None,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




