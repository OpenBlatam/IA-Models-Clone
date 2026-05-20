"""
Progress tracking routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

try:
    from core.progress_tracker import ProgressTracker
except ImportError:
    from ...core.progress_tracker import ProgressTracker

router = APIRouter()

tracker = ProgressTracker()


class LogEntryRequest(BaseModel):
    user_id: str
    date: str
    mood: str
    cravings_level: int
    triggers_encountered: List[str] = []
    consumed: bool = False
    notes: Optional[str] = None


@router.post("/log-entry")
async def log_progress_entry(request: LogEntryRequest):
    """Registra una entrada de progreso"""
    try:
        entry_data = request.dict()
        entry = tracker.log_entry(entry_data)
        return JSONResponse(content=entry)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registrando entrada: {str(e)}")


@router.get("/progress/{user_id}")
async def get_progress(
    user_id: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """Obtiene progreso del usuario"""
    try:
        progress = tracker.get_progress(user_id, start_date, end_date)
        return JSONResponse(content=progress)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo progreso: {str(e)}")


@router.get("/stats/{user_id}")
async def get_progress_stats(user_id: str):
    """Obtiene estadísticas de progreso"""
    try:
        stats = tracker.get_statistics(user_id)
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")


@router.get("/timeline/{user_id}")
async def get_progress_timeline(
    user_id: str,
    days: Optional[int] = Query(30)
):
    """Obtiene línea de tiempo de progreso"""
    try:
        timeline = tracker.get_timeline(user_id, days)
        return JSONResponse(content=timeline)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo timeline: {str(e)}")



