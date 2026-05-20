"""
Challenges endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.challenges import ChallengesService, ChallengeType

router = APIRouter()
challenges_service = ChallengesService()


@router.get("/available/{user_id}")
async def get_available_challenges(
    user_id: str,
    challenge_type: Optional[str] = None
) -> Dict[str, Any]:
    """Obtener desafíos disponibles"""
    try:
        challenge_type_enum = ChallengeType(challenge_type) if challenge_type else None
        challenges = challenges_service.get_available_challenges(user_id, challenge_type_enum)
        
        return {
            "challenges": [
                {
                    "id": c.id,
                    "title": c.title,
                    "description": c.description,
                    "type": c.type.value,
                    "difficulty": c.difficulty,
                    "points_reward": c.points_reward,
                    "xp_reward": c.xp_reward,
                    "badge_reward": c.badge_reward,
                    "status": c.status.value,
                    "progress": c.progress,
                }
                for c in challenges
            ],
            "total": len(challenges),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start/{user_id}/{challenge_id}")
async def start_challenge(user_id: str, challenge_id: str) -> Dict[str, Any]:
    """Iniciar un desafío"""
    try:
        challenge = challenges_service.start_challenge(user_id, challenge_id)
        return {
            "challenge_id": challenge.id,
            "status": challenge.status.value,
            "started_at": challenge.started_at.isoformat() if challenge.started_at else None,
            "expires_at": challenge.expires_at.isoformat() if challenge.expires_at else None,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/progress/{user_id}/{challenge_id}")
async def update_progress(
    user_id: str,
    challenge_id: str,
    progress: float
) -> Dict[str, Any]:
    """Actualizar progreso de desafío"""
    try:
        challenge = challenges_service.update_challenge_progress(
            user_id, challenge_id, progress
        )
        return {
            "challenge_id": challenge.id,
            "progress": challenge.progress,
            "status": challenge.status.value,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complete/{user_id}/{challenge_id}")
async def complete_challenge(user_id: str, challenge_id: str) -> Dict[str, Any]:
    """Completar un desafío"""
    try:
        result = challenges_service.complete_challenge(user_id, challenge_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




