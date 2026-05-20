"""
Export API Routes
=================

Endpoints para exportar datos.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from typing import Optional
import tempfile
import os

router = APIRouter(prefix="/export", tags=["export"])


def get_community_manager():
    """Dependency para obtener CommunityManager"""
    from ...core.community_manager import CommunityManager
    return CommunityManager()


@router.get("/posts/csv")
async def export_posts_csv(
    status: Optional[str] = Query(None),
    manager = Depends(get_community_manager)
):
    """Exportar posts a CSV"""
    try:
        from ...utils.export_utils import ExportUtils
        
        posts = manager.scheduler.get_all_posts(status=status)
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            file_path = f.name
        
        success = ExportUtils.export_posts_to_csv(posts, file_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Error exportando posts")
        
        return FileResponse(
            file_path,
            media_type="text/csv",
            filename="posts_export.csv",
            background=lambda: os.unlink(file_path)  # Limpiar después
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calendar/ical")
async def export_calendar_ical(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    manager = Depends(get_community_manager)
):
    """Exportar calendario a iCal"""
    try:
        from ...utils.export_utils import ExportUtils
        from datetime import datetime
        
        if start_date:
            start = datetime.fromisoformat(start_date)
        else:
            start = None
        
        if end_date:
            end = datetime.fromisoformat(end_date)
        else:
            end = None
        
        events = manager.get_calendar_view(start, end)
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ics') as f:
            file_path = f.name
        
        success = ExportUtils.export_calendar_to_ical(events, file_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Error exportando calendario")
        
        return FileResponse(
            file_path,
            media_type="text/calendar",
            filename="calendar.ics",
            background=lambda: os.unlink(file_path)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
async def generate_report(
    manager = Depends(get_community_manager)
):
    """Generar reporte completo"""
    try:
        from ...utils.export_utils import ExportUtils
        
        posts = manager.scheduler.get_all_posts()
        analytics = manager.get_analytics()
        
        report_content = ExportUtils.generate_report(posts, analytics)
        
        return {"report": report_content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




