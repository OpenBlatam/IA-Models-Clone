"""
Challenges routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.challenge_service import ChallengeService
except ImportError:
    from ...services.challenge_service import ChallengeService

router = APIRouter()

challenges = ChallengeService()


@router.post("/challenges/create")
async def create_challenge(
    user_id: str = Body(...),
    challenge_type: str = Body(...),
    title: str = Body(...),
    description: str = Body(...),
    duration_days: int = Body(...)
):
    """Crea un nuevo desafío"""
    try:
        challenge = challenges.create_challenge(
            user_id, challenge_type, title, description, duration_days
        )
        return JSONResponse(content=challenge)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando desafío: {str(e)}")


@router.get("/challenges/available/{user_id}")
async def get_available_challenges(
    user_id: str,
    challenge_type: Optional[str] = Query(None)
):
    """Obtiene desafíos disponibles"""
    try:
        available = challenges.get_available_challenges(user_id, challenge_type)
        return JSONResponse(content={
            "user_id": user_id,
            "challenges": available,
            "total": len(available),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo desafíos: {str(e)}")



