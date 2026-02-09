"""
Feedback endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.feedback import FeedbackService, FeedbackType, FeedbackStatus

router = APIRouter()
feedback_service = FeedbackService()


@router.post("/submit/{user_id}")
async def submit_feedback(
    user_id: str,
    feedback_type: str,
    title: str,
    description: str,
    rating: Optional[int] = None
) -> Dict[str, Any]:
    """Enviar feedback"""
    try:
        type_enum = FeedbackType(feedback_type)
        feedback = feedback_service.submit_feedback(
            user_id, type_enum, title, description, rating
        )
        return {
            "id": feedback.id,
            "status": feedback.status.value,
            "created_at": feedback.created_at.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def get_feedback(
    feedback_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """Obtener feedback"""
    try:
        type_enum = FeedbackType(feedback_type) if feedback_type else None
        status_enum = FeedbackStatus(status) if status else None
        
        feedback = feedback_service.get_feedback(type_enum, status_enum, limit)
        return {
            "feedback": [
                {
                    "id": f.id,
                    "type": f.feedback_type.value,
                    "title": f.title,
                    "rating": f.rating,
                    "status": f.status.value,
                    "created_at": f.created_at.isoformat(),
                }
                for f in feedback
            ],
            "total": len(feedback),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_feedback_stats() -> Dict[str, Any]:
    """Obtener estadísticas de feedback"""
    try:
        stats = feedback_service.get_feedback_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




