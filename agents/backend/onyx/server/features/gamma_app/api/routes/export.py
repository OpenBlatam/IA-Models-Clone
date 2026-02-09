"""
Export Routes
API endpoints for exporting content in various formats
"""

import logging
import io
from typing import Dict, Any

from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse

from ..models import ExportRequest, User
from ..dependencies import get_analytics_service, get_current_user
from ..error_handlers import handle_route_errors
from ...engines.presentation_engine import PresentationEngine
from ...engines.document_engine import DocumentEngine
from ...services.analytics_service import AnalyticsService
from ..tasks.analytics_tasks import track_export

logger = logging.getLogger(__name__)

router = APIRouter()

def _create_streaming_response(
    content: bytes,
    media_type: str,
    filename: str,
    file_extension: str
) -> StreamingResponse:
    """Helper to create streaming response"""
    return StreamingResponse(
        io.BytesIO(content),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}.{file_extension}"}
    )

@router.post("/presentation")
@handle_route_errors
async def export_presentation(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: User = Depends(get_current_user)
) -> StreamingResponse:
    """Export presentation in specified format"""
    presentation_engine = PresentationEngine()
    presentation_bytes = await presentation_engine.create_presentation(
        content=request.content,
        theme=request.theme or "modern",
        template=request.template or "business_pitch"
    )
    
    background_tasks.add_task(
        track_export,
        current_user.id,
        "presentation",
        request.output_format,
        analytics_service
    )
    
    return _create_streaming_response(
        presentation_bytes,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "presentation",
        request.output_format
    )

@router.post("/document")
@handle_route_errors
async def export_document(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: User = Depends(get_current_user)
) -> StreamingResponse:
    """Export document in specified format"""
    document_engine = DocumentEngine()
    document_bytes = await document_engine.create_document(
        content=request.content,
        doc_type=request.document_type or "report",
        style=request.style or "business",
        output_format=request.output_format
    )
    
    background_tasks.add_task(
        track_export,
        current_user.id,
        "document",
        request.output_format,
        analytics_service
    )
    
    return _create_streaming_response(
        document_bytes,
        "application/octet-stream",
        "document",
        request.output_format
    )

@router.post("/webpage")
@handle_route_errors
async def export_webpage(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: User = Depends(get_current_user)
) -> StreamingResponse:
    """Export webpage in specified format"""
    title = request.content.get('title', 'Web Page')
    description = request.content.get('description', 'Generated web page content')
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        p {{ line-height: 1.6; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>{description}</p>
</body>
</html>"""
    
    background_tasks.add_task(
        track_export,
        current_user.id,
        "webpage",
        request.output_format,
        analytics_service
    )
    
    return _create_streaming_response(
        html_content.encode('utf-8'),
        "text/html",
        "webpage",
        request.output_format
    )







