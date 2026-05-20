"""
Export routes for Lovable Community SAM3.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime
import logging

from ...database import get_db_session
from ...services.export_service import ExportService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/export", tags=["export"])


@router.get("/chats/{chat_id}")
async def export_chat(
    chat_id: str,
    format: str = Query("json", pattern="^(json|csv)$", description="Export format"),
    include_stats: bool = Query(True, description="Include statistics"),
    include_comments: bool = Query(False, description="Include comments"),
    include_votes: bool = Query(False, description="Include votes"),
    db: Session = Depends(get_db_session)
):
    """Export a chat with optional related data."""
    export_service = ExportService(db)
    
    if format == "csv":
        csv_data = export_service.export_chat_csv(chat_id)
        return JSONResponse(
            content={"csv": csv_data},
            media_type="application/json"
        )
    
    export_data = export_service.export_chat(
        chat_id=chat_id,
        include_stats=include_stats,
        include_comments=include_comments,
        include_votes=include_votes
    )
    
    return export_data


@router.get("/users/{user_id}")
async def export_user_data(
    user_id: str,
    format: str = Query("json", pattern="^(json|csv)$", description="Export format"),
    include_chats: bool = Query(True, description="Include chats"),
    include_comments: bool = Query(True, description="Include comments"),
    include_bookmarks: bool = Query(True, description="Include bookmarks"),
    db: Session = Depends(get_db_session)
):
    """Export all user data."""
    export_service = ExportService(db)
    export_data = export_service.export_user_data(
        user_id=user_id,
        include_chats=include_chats,
        include_comments=include_comments,
        include_bookmarks=include_bookmarks
    )
    
    return export_data


@router.get("/analytics/summary")
async def export_analytics_summary(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    db: Session = Depends(get_db_session)
):
    """Export analytics summary."""
    export_service = ExportService(db)
    
    # Parse dates with error handling
    start = None
    end = None
    
    if start_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid start_date format. Expected YYYY-MM-DD, got: {start_date}"
            )
    
    if end_date:
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid end_date format. Expected YYYY-MM-DD, got: {end_date}"
            )
    
    summary = export_service.export_analytics_summary(
        start_date=start,
        end_date=end
    )
    
    return summary







