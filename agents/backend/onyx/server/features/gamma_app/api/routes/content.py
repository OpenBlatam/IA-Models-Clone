"""
Content Generation Routes
API endpoints for content generation and management
"""

import logging
from typing import List, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, BackgroundTasks, Query, Path

from ..models import ContentRequest, ContentResponse, User
from ..dependencies import get_content_generator, get_analytics_service, get_current_user
from ..error_handlers import handle_route_errors
from ...core.content_generator import ContentGenerator, ContentType, OutputFormat, DesignStyle
from ...services.analytics_service import AnalyticsService
from ..tasks.analytics_tasks import track_content_generation

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/generate", response_model=ContentResponse)
@handle_route_errors
async def generate_content(
    request: ContentRequest,
    background_tasks: BackgroundTasks,
    generator: ContentGenerator = Depends(get_content_generator),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: User = Depends(get_current_user)
) -> ContentResponse:
    """Generate new content based on request"""
    request.user_id = current_user.id
    request.project_id = request.project_id or str(uuid4())
    response = await generator.generate_content(request)
    
    background_tasks.add_task(
        track_content_generation,
        current_user.id,
        request.content_type.value,
        response.processing_time,
        analytics_service
    )
    
    return response

@router.get("/types", response_model=List[Dict[str, Any]])
async def get_content_types() -> List[Dict[str, Any]]:
    """Get available content types"""
    return [
        {"value": ct.value, "label": ct.value.replace("_", " ").title()}
        for ct in ContentType
    ]

@router.get("/formats", response_model=List[Dict[str, Any]])
async def get_output_formats() -> List[Dict[str, Any]]:
    """Get available output formats"""
    return [
        {"value": of.value, "label": of.value.upper()}
        for of in OutputFormat
    ]

@router.get("/styles", response_model=List[Dict[str, Any]])
async def get_design_styles() -> List[Dict[str, Any]]:
    """Get available design styles"""
    return [
        {"value": ds.value, "label": ds.value.replace("_", " ").title()}
        for ds in DesignStyle
    ]

@router.get("/templates/{content_type}")
@handle_route_errors
async def get_templates(
    content_type: ContentType = Path(..., description="Content type"),
    generator: ContentGenerator = Depends(get_content_generator)
) -> Dict[str, Any]:
    """Get available templates for content type"""
    templates = generator.get_available_templates(content_type)
    return {"templates": templates}

@router.post("/enhance/{content_id}")
@handle_route_errors
async def enhance_content(
    content_id: str = Path(..., description="Content ID"),
    enhancement_type: str = Query(..., description="Enhancement type"),
    instructions: str = Query(..., description="Enhancement instructions"),
    generator: ContentGenerator = Depends(get_content_generator),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Enhance existing content"""
    response = await generator.enhance_content(
        content_id, enhancement_type, instructions
    )
    return response

@router.get("/suggestions/{content_id}")
@handle_route_errors
async def get_content_suggestions(
    content_id: str = Path(..., description="Content ID"),
    generator: ContentGenerator = Depends(get_content_generator),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get suggestions for improving content"""
    suggestions = await generator.get_content_suggestions(content_id)
    return {"suggestions": suggestions}







