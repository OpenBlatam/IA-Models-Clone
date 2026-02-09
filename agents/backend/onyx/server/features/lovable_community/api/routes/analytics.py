"""
Rutas para analytics y perfiles

Incluye: get_analytics, get_user_profile
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from ...dependencies import get_chat_service
from ...schemas import AnalyticsResponse, UserProfileResponse
from ...services import ChatService
from ...helpers.validation_common import ensure_not_empty_string
from ..cache import cache_response
from ..decorators import handle_errors, log_request, measure_time

router = APIRouter()


@router.get(
    "/analytics",
    response_model=AnalyticsResponse,
    summary="Obtener analytics de la comunidad",
    description="Obtiene estadísticas agregadas de toda la comunidad"
)
@cache_response(ttl=300.0)
@log_request
@measure_time
@handle_errors
async def get_analytics(
    period_days: Optional[int] = Query(None, ge=1, le=365, description="Número de días para filtrar (1-365, opcional)"),
    service: ChatService = Depends(get_chat_service)
) -> AnalyticsResponse:
    """
    Get community analytics and aggregated statistics.
    
    Args:
        period_days: Optional number of days to filter (1-365). If None, returns all-time stats
        service: ChatService instance
        
    Returns:
        AnalyticsResponse with community statistics
        
    Raises:
        ValueError: If period_days is out of valid range
    """
    # Validate period_days if provided
    if period_days is not None:
        if period_days < 1:
            raise ValueError(f"period_days must be >= 1, got {period_days}")
        if period_days > 365:
            raise ValueError(f"period_days must be <= 365, got {period_days}")
    
    analytics = service.get_analytics(period_days)
    period_str = f"{period_days} days" if period_days else "all time"
    
    return AnalyticsResponse(
        **analytics,
        period=period_str
    )


@router.get(
    "/users/{user_id}/profile",
    response_model=UserProfileResponse,
    summary="Obtener perfil de usuario",
    description="Obtiene el perfil y estadísticas de un usuario"
)
@cache_response(ttl=120.0)
@log_request
@handle_errors
async def get_user_profile(
    user_id: str,
    service: ChatService = Depends(get_chat_service)
) -> UserProfileResponse:
    """
    Get user profile and statistics.
    
    Args:
        user_id: User ID to get profile for
        service: ChatService instance
        
    Returns:
        UserProfileResponse with user statistics
        
    Raises:
        ValueError: If user_id is None or empty
    """
    user_id = ensure_not_empty_string(user_id, "user_id")
    profile = service.get_user_profile(user_id)
    return UserProfileResponse(**profile)

