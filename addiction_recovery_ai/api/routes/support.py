"""
Support and motivation routes - Refactored with best practices
"""

from fastapi import APIRouter, HTTPException, status

try:
    from schemas.support import (
        CoachingSessionRequest,
        CoachingSessionResponse,
        MotivationResponse,
        MilestoneRequest,
        MilestoneResponse,
        AchievementsResponse
    )
    from schemas.common import ErrorResponse
    from dependencies import (
        CounselingServiceDep,
        MotivationServiceDep
    )
except ImportError:
    from ...schemas.support import (
        CoachingSessionRequest,
        CoachingSessionResponse,
        MotivationResponse,
        MilestoneRequest,
        MilestoneResponse,
        AchievementsResponse
    )
    from ...schemas.common import ErrorResponse
    from ...dependencies import (
        CounselingServiceDep,
        MotivationServiceDep
    )

router = APIRouter(prefix="/support", tags=["Support & Motivation"])


@router.post(
    "/coaching-session",
    response_model=CoachingSessionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def coaching_session(
    request: CoachingSessionRequest,
    counseling: CounselingServiceDep
) -> CoachingSessionResponse:
    """
    Sesión de coaching personalizado
    
    - **user_id**: ID del usuario
    - **topic**: Tema de la sesión
    - **current_situation**: Descripción de la situación actual
    - **questions**: Preguntas específicas (opcional)
    """
    # Guard clause: Validate required fields
    if not request.topic or not request.topic.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="topic es requerido"
        )
    
    if not request.current_situation or not request.current_situation.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="current_situation es requerido"
        )
    
    try:
        session = counseling.create_coaching_session(
            request.user_id,
            request.topic,
            request.current_situation,
            request.questions
        )
        
        return CoachingSessionResponse(
            user_id=request.user_id,
            session_id=session.get("session_id", ""),
            topic=request.topic,
            guidance=session.get("guidance", ""),
            questions_to_consider=session.get("questions_to_consider", []),
            action_items=session.get("action_items", []),
            encouragement=session.get("encouragement", "")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en sesión de coaching: {str(e)}"
        )


@router.get(
    "/motivation/{user_id}",
    response_model=MotivationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_motivation(
    user_id: str,
    motivation: MotivationServiceDep
) -> MotivationResponse:
    """
    Obtiene mensajes motivacionales personalizados
    
    - **user_id**: ID del usuario
    """
    # Guard clause: Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    try:
        messages = motivation.get_motivational_messages(user_id, {})
        
        return MotivationResponse(
            user_id=user_id,
            messages=messages.get("messages", []),
            personalized_message=messages.get("personalized_message"),
            achievements_summary=messages.get("achievements_summary")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo motivación: {str(e)}"
        )


@router.post(
    "/celebrate-milestone",
    response_model=MilestoneResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def celebrate_milestone(
    request: MilestoneRequest,
    motivation: MotivationServiceDep
) -> MilestoneResponse:
    """
    Celebra un logro/hito
    
    - **user_id**: ID del usuario
    - **milestone_days**: Días del hito a celebrar
    """
    # Guard clause: Validate milestone_days
    if request.milestone_days < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="milestone_days debe ser mayor a 0"
        )
    
    try:
        celebration = motivation.celebrate_milestone(
            request.user_id,
            request.milestone_days
        )
        
        return MilestoneResponse(
            user_id=request.user_id,
            milestone_days=request.milestone_days,
            celebration_message=celebration.get("message", ""),
            rewards=celebration.get("rewards", []),
            next_milestone=celebration.get("next_milestone")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error celebrando hito: {str(e)}"
        )


@router.get(
    "/achievements/{user_id}",
    response_model=AchievementsResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_achievements(
    user_id: str,
    motivation: MotivationServiceDep
) -> AchievementsResponse:
    """
    Obtiene logros del usuario
    
    - **user_id**: ID del usuario
    """
    # Guard clause: Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    try:
        achievements = motivation.get_achievements(user_id)
        
        return AchievementsResponse(
            user_id=user_id,
            achievements=achievements.get("achievements", []),
            total_points=achievements.get("total_points", 0),
            level=achievements.get("level")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo logros: {str(e)}"
        )

