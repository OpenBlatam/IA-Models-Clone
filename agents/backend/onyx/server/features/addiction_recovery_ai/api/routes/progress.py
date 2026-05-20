"""
Progress tracking routes - Refactored with best practices
"""

from fastapi import APIRouter, HTTPException, Query, status
from typing import Optional
from datetime import datetime

try:
    from schemas.progress import (
        LogEntryRequest,
        LogEntryResponse,
        ProgressResponse,
        StatsResponse,
        TimelineResponse
    )
    from schemas.common import ErrorResponse
    from dependencies import ProgressTrackerDep
except ImportError:
    from ...schemas.progress import (
        LogEntryRequest,
        LogEntryResponse,
        ProgressResponse,
        StatsResponse,
        TimelineResponse
    )
    from ...schemas.common import ErrorResponse
    from ...dependencies import ProgressTrackerDep

router = APIRouter(prefix="/progress", tags=["Progress"])


@router.post(
    "/log-entry",
    response_model=LogEntryResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def log_daily_entry(
    request: LogEntryRequest,
    tracker: ProgressTrackerDep
) -> LogEntryResponse:
    """
    Registra una entrada diaria de progreso
    
    - **user_id**: ID del usuario
    - **date**: Fecha de la entrada (ISO format)
    - **mood**: Estado de ánimo (excellent, good, neutral, poor, terrible)
    - **cravings_level**: Nivel de antojos (0-10)
    - **triggers_encountered**: Lista de triggers encontrados
    - **consumed**: Si se consumió la sustancia
    - **notes**: Notas adicionales
    """
    # Guard clause: Validate date format
    try:
        entry_date = datetime.fromisoformat(request.date)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato de fecha inválido. Use formato ISO (YYYY-MM-DD)"
        )
    
    # Guard clause: Validate date is not in the future
    if entry_date > datetime.now():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La fecha no puede ser en el futuro"
        )
    
    # Process entry
    try:
        entry = tracker.log_entry(
            request.user_id,
            request.date,
            request.mood,
            request.cravings_level,
            request.triggers_encountered,
            request.consumed,
            request.notes
        )
        
        return LogEntryResponse(
            entry_id=entry.get("entry_id", ""),
            user_id=request.user_id,
            date=request.date,
            mood=request.mood,
            cravings_level=request.cravings_level,
            triggers_encountered=request.triggers_encountered,
            consumed=request.consumed,
            notes=request.notes
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registrando entrada: {str(e)}"
        )


@router.get(
    "/{user_id}",
    response_model=ProgressResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_progress(
    user_id: str,
    start_date: Optional[str] = Query(None, description="Start date for progress (ISO format)"),
    tracker: ProgressTrackerDep
) -> ProgressResponse:
    """
    Obtiene progreso del usuario
    
    - **user_id**: ID del usuario
    - **start_date**: Fecha de inicio opcional (ISO format)
    """
    # Guard clause: Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    # Parse start_date if provided
    start = None
    if start_date:
        try:
            start = datetime.fromisoformat(start_date)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Formato de fecha inválido. Use formato ISO (YYYY-MM-DD)"
            )
    
    # Get progress
    try:
        progress = tracker.get_progress(user_id, start, [])
        
        return ProgressResponse(
            user_id=user_id,
            days_sober=progress.get("days_sober", 0),
            total_entries=progress.get("total_entries", 0),
            streak_days=progress.get("streak_days", 0),
            longest_streak=progress.get("longest_streak", 0),
            progress_percentage=progress.get("progress_percentage", 0.0),
            recent_entries=progress.get("recent_entries", [])
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo progreso: {str(e)}"
        )


@router.get(
    "/{user_id}/stats",
    response_model=StatsResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_stats(
    user_id: str,
    tracker: ProgressTrackerDep
) -> StatsResponse:
    """
    Obtiene estadísticas detalladas del usuario
    
    - **user_id**: ID del usuario
    """
    # Guard clause: Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    try:
        stats = tracker.get_stats(user_id, [])
        
        return StatsResponse(
            user_id=user_id,
            total_days=stats.get("total_days", 0),
            days_sober=stats.get("days_sober", 0),
            relapse_count=stats.get("relapse_count", 0),
            average_cravings=stats.get("average_cravings", 0.0),
            most_common_triggers=stats.get("most_common_triggers", []),
            trends=stats.get("trends", {})
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo estadísticas: {str(e)}"
        )


@router.get(
    "/{user_id}/timeline",
    response_model=TimelineResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_timeline(
    user_id: str,
    tracker: ProgressTrackerDep
) -> TimelineResponse:
    """
    Obtiene línea de tiempo de progreso
    
    - **user_id**: ID del usuario
    """
    # Guard clause: Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    try:
        timeline = tracker.get_timeline(user_id, [])
        
        return TimelineResponse(
            user_id=user_id,
            timeline=timeline.get("timeline", []),
            milestones=timeline.get("milestones", []),
            relapses=timeline.get("relapses", [])
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo timeline: {str(e)}"
        )

