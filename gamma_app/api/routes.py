"""
Gamma App - API Routes
RESTful API endpoints for content generation and collaboration
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import asyncio

from .models import (
    ContentRequest, ContentResponse, User, Project, 
    CollaborationSession, ExportRequest, AnalyticsRequest
)
from ..core.content_generator import ContentGenerator, ContentType, OutputFormat, DesignStyle
from ..engines.presentation_engine import PresentationEngine
from ..engines.document_engine import DocumentEngine
from ..services.collaboration_service import CollaborationService
from ..services.analytics_service import AnalyticsService
from ..utils.auth import get_current_user

logger = logging.getLogger(__name__)

# Create routers
content_router = APIRouter()
collaboration_router = APIRouter()
export_router = APIRouter()
analytics_router = APIRouter()

# Content Generation Routes
@content_router.post("/generate", response_model=ContentResponse)
async def generate_content(
    request: ContentRequest,
    background_tasks: BackgroundTasks,
    generator: ContentGenerator = Depends(lambda: None),  # Will be injected
    current_user: User = Depends(get_current_user)
):
    """Generate new content based on request"""
    try:
        # Add user context to request
        request.user_id = current_user.id
        request.project_id = request.project_id or str(uuid4())
        
        # Generate content
        response = await generator.generate_content(request)
        
        # Track analytics in background
        background_tasks.add_task(
            track_content_generation,
            current_user.id,
            request.content_type.value,
            response.processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@content_router.get("/types", response_model=List[Dict[str, Any]])
async def get_content_types():
    """Get available content types"""
    return [
        {"value": ct.value, "label": ct.value.replace("_", " ").title()}
        for ct in ContentType
    ]

@content_router.get("/formats", response_model=List[Dict[str, Any]])
async def get_output_formats():
    """Get available output formats"""
    return [
        {"value": of.value, "label": of.value.upper()}
        for of in OutputFormat
    ]

@content_router.get("/styles", response_model=List[Dict[str, Any]])
async def get_design_styles():
    """Get available design styles"""
    return [
        {"value": ds.value, "label": ds.value.replace("_", " ").title()}
        for ds in DesignStyle
    ]

@content_router.get("/templates/{content_type}")
async def get_templates(
    content_type: ContentType = Path(..., description="Content type"),
    generator: ContentGenerator = Depends(lambda: None)
):
    """Get available templates for content type"""
    try:
        templates = generator.get_available_templates(content_type)
        return {"templates": templates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@content_router.post("/enhance/{content_id}")
async def enhance_content(
    content_id: str = Path(..., description="Content ID"),
    enhancement_type: str = Query(..., description="Enhancement type"),
    instructions: str = Query(..., description="Enhancement instructions"),
    generator: ContentGenerator = Depends(lambda: None),
    current_user: User = Depends(get_current_user)
):
    """Enhance existing content"""
    try:
        response = await generator.enhance_content(
            content_id, enhancement_type, instructions
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@content_router.get("/suggestions/{content_id}")
async def get_content_suggestions(
    content_id: str = Path(..., description="Content ID"),
    generator: ContentGenerator = Depends(lambda: None),
    current_user: User = Depends(get_current_user)
):
    """Get suggestions for improving content"""
    try:
        suggestions = await generator.get_content_suggestions(content_id)
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Collaboration Routes
@collaboration_router.post("/session", response_model=CollaborationSession)
async def create_collaboration_session(
    project_id: str = Query(..., description="Project ID"),
    session_name: str = Query(..., description="Session name"),
    collaboration_service: CollaborationService = Depends(lambda: None),
    current_user: User = Depends(get_current_user)
):
    """Create new collaboration session"""
    try:
        session = await collaboration_service.create_session(
            project_id=project_id,
            session_name=session_name,
            creator_id=current_user.id
        )
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@collaboration_router.get("/session/{session_id}")
async def get_collaboration_session(
    session_id: str = Path(..., description="Session ID"),
    collaboration_service: CollaborationService = Depends(lambda: None),
    current_user: User = Depends(get_current_user)
):
    """Get collaboration session details"""
    try:
        session = await collaboration_service.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@collaboration_router.post("/session/{session_id}/join")
async def join_collaboration_session(
    session_id: str = Path(..., description="Session ID"),
    collaboration_service: CollaborationService = Depends(lambda: None),
    current_user: User = Depends(get_current_user)
):
    """Join collaboration session"""
    try:
        success = await collaboration_service.join_session(
            session_id=session_id,
            user_id=current_user.id
        )
        if not success:
            raise HTTPException(status_code=400, detail="Could not join session")
        return {"message": "Successfully joined session"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@collaboration_router.post("/session/{session_id}/leave")
async def leave_collaboration_session(
    session_id: str = Path(..., description="Session ID"),
    collaboration_service: CollaborationService = Depends(lambda: None),
    current_user: User = Depends(get_current_user)
):
    """Leave collaboration session"""
    try:
        success = await collaboration_service.leave_session(
            session_id=session_id,
            user_id=current_user.id
        )
        if not success:
            raise HTTPException(status_code=400, detail="Could not leave session")
        return {"message": "Successfully left session"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@collaboration_router.get("/session/{session_id}/participants")
async def get_session_participants(
    session_id: str = Path(..., description="Session ID"),
    collaboration_service: CollaborationService = Depends(lambda: None),
    current_user: User = Depends(get_current_user)
):
    """Get session participants"""
    try:
        participants = await collaboration_service.get_session_participants(session_id)
        return {"participants": participants}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@collaboration_router.websocket("/session/{session_id}/ws")
async def collaboration_websocket(
    websocket,
    session_id: str,
    collaboration_service: CollaborationService = Depends(lambda: None)
):
    """WebSocket endpoint for real-time collaboration"""
    await collaboration_service.handle_websocket_connection(websocket, session_id)

# Export Routes
@export_router.post("/presentation")
async def export_presentation(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Export presentation in specified format"""
    try:
        # Create presentation engine
        presentation_engine = PresentationEngine()
        
        # Generate presentation
        presentation_bytes = await presentation_engine.create_presentation(
            content=request.content,
            theme=request.theme or "modern",
            template=request.template or "business_pitch"
        )
        
        # Track export in background
        background_tasks.add_task(
            track_export,
            current_user.id,
            "presentation",
            request.output_format
        )
        
        # Return file
        return StreamingResponse(
            io.BytesIO(presentation_bytes),
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": f"attachment; filename=presentation.{request.output_format}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@export_router.post("/document")
async def export_document(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Export document in specified format"""
    try:
        # Create document engine
        document_engine = DocumentEngine()
        
        # Generate document
        document_bytes = await document_engine.create_document(
            content=request.content,
            doc_type=request.document_type or "report",
            style=request.style or "business",
            output_format=request.output_format
        )
        
        # Track export in background
        background_tasks.add_task(
            track_export,
            current_user.id,
            "document",
            request.output_format
        )
        
        # Return file
        return StreamingResponse(
            io.BytesIO(document_bytes),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename=document.{request.output_format}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@export_router.post("/webpage")
async def export_webpage(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Export webpage in specified format"""
    try:
        # Create web page engine (would be implemented)
        # For now, return HTML content
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{request.content.get('title', 'Web Page')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                p {{ line-height: 1.6; }}
            </style>
        </head>
        <body>
            <h1>{request.content.get('title', 'Web Page')}</h1>
            <p>{request.content.get('description', 'Generated web page content')}</p>
        </body>
        </html>
        """
        
        # Track export in background
        background_tasks.add_task(
            track_export,
            current_user.id,
            "webpage",
            request.output_format
        )
        
        # Return file
        return StreamingResponse(
            io.BytesIO(html_content.encode('utf-8')),
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename=webpage.{request.output_format}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Analytics Routes
@analytics_router.get("/dashboard")
async def get_analytics_dashboard(
    time_period: str = Query("7d", description="Time period"),
    analytics_service: AnalyticsService = Depends(lambda: None),
    current_user: User = Depends(get_current_user)
):
    """Get analytics dashboard data"""
    try:
        dashboard_data = await analytics_service.get_dashboard_data(
            user_id=current_user.id,
            time_period=time_period
        )
        return dashboard_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@analytics_router.get("/content-performance")
async def get_content_performance(
    content_id: Optional[str] = Query(None, description="Content ID"),
    time_period: str = Query("30d", description="Time period"),
    analytics_service: AnalyticsService = Depends(lambda: None),
    current_user: User = Depends(get_current_user)
):
    """Get content performance analytics"""
    try:
        performance_data = await analytics_service.get_content_performance(
            user_id=current_user.id,
            content_id=content_id,
            time_period=time_period
        )
        return performance_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@analytics_router.get("/collaboration-stats")
async def get_collaboration_stats(
    time_period: str = Query("30d", description="Time period"),
    analytics_service: AnalyticsService = Depends(lambda: None),
    current_user: User = Depends(get_current_user)
):
    """Get collaboration statistics"""
    try:
        stats = await analytics_service.get_collaboration_stats(
            user_id=current_user.id,
            time_period=time_period
        )
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
async def track_content_generation(user_id: str, content_type: str, processing_time: float):
    """Track content generation analytics"""
    # This would integrate with analytics service
    logger.info(f"Tracking content generation: {user_id}, {content_type}, {processing_time}s")

async def track_export(user_id: str, content_type: str, output_format: str):
    """Track export analytics"""
    # This would integrate with analytics service
    logger.info(f"Tracking export: {user_id}, {content_type}, {output_format}")

# Import io for file operations
import io



























