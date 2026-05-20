"""
Calendar API Routes
===================

Endpoints para gestión del calendario de publicaciones.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

router = APIRouter(prefix="/calendar", tags=["calendar"])


class CalendarEventResponse(BaseModel):
    id: str
    scheduled_time: str
    content: str
    platforms: List[str]
    status: str


def get_community_manager():
    """Dependency para obtener CommunityManager"""
    from ...core.community_manager import CommunityManager
    return CommunityManager()


@router.get("/", response_model=List[dict])
async def get_calendar(
    start_date: Optional[datetime] = Query(None, description="Fecha de inicio"),
    end_date: Optional[datetime] = Query(None, description="Fecha de fin"),
    manager = Depends(get_community_manager)
):
    """Obtener eventos del calendario en un rango de fechas"""
    try:
        events = manager.get_calendar_view(start_date, end_date)
        return events
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily", response_model=List[dict])
async def get_daily_events(
    date: datetime = Query(..., description="Fecha del día"),
    manager = Depends(get_community_manager)
):
    """Obtener eventos de un día específico"""
    try:
        events = manager.calendar.get_daily_events(date)
        return events
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weekly", response_model=dict)
async def get_weekly_view(
    start_date: Optional[datetime] = Query(None, description="Fecha de inicio de semana"),
    manager = Depends(get_community_manager)
):
    """Obtener vista semanal del calendario"""
    try:
        weekly_view = manager.calendar.get_weekly_view(start_date)
        return weekly_view
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




