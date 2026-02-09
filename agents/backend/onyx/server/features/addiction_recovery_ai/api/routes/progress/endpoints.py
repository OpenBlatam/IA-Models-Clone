"""
Progress tracking endpoints - Ultra modular implementation
"""

from fastapi import APIRouter, Query, status
from typing import Optional

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
    from utils.validators import validate_user_id
    from .validators import validate_log_entry_request
    from .handlers import (
        create_log_entry,
        get_user_progress,
        get_user_stats,
        get_user_timeline
    )
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
    from ...utils.validators import validate_user_id
    from .validators import validate_log_entry_request
    from .handlers import (
        create_log_entry,
        get_user_progress,
        get_user_stats,
        get_user_timeline
    )

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
    """
    await validate_log_entry_request(request)
    return await create_log_entry(request, tracker)


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
    tracker: ProgressTrackerDep = None
) -> ProgressResponse:
    """
    Obtiene progreso del usuario
    
    - **user_id**: ID del usuario
    - **start_date**: Fecha de inicio opcional (ISO format)
    """
    validate_user_id(user_id)
    return await get_user_progress(user_id, start_date, tracker)


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
    validate_user_id(user_id)
    return await get_user_stats(user_id, tracker)


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
    validate_user_id(user_id)
    return await get_user_timeline(user_id, tracker)

