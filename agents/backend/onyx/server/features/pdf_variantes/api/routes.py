"""
PDF Variantes API Routes
Comprehensive REST API endpoints for PDF processing with AI capabilities
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Path, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ..models import (
    PDFUploadRequest, PDFUploadResponse, PDFDocument,
    VariantGenerateRequest, VariantGenerateResponse, PDFVariant,
    TopicExtractRequest, TopicExtractResponse, TopicItem,
    BrainstormGenerateRequest, BrainstormGenerateResponse, BrainstormIdea,
    ExportRequest, ExportResponse,
    CollaborationInvite, Annotation, Feedback,
    SearchRequest, SearchResponse, SearchResult,
    BatchProcessingRequest, BatchProcessingResponse,
    SystemHealth, AnalyticsReport
)
from ..services.pdf_service import PDFVariantesService
from ..services.cache_service import CacheService
from ..services.security_service import SecurityService
from ..services.analytics_service import AnalyticsService
from ..services.collaboration_service import CollaborationService
from ..services.notification_service import NotificationService
from ..utils.auth import get_current_user, require_permissions
from ..utils.validators import validate_file_upload, validate_content_type
from ..utils.validation import (
    Validator,
    validate_filename,
    validate_file_extension,
    validate_integer_range,
    validate_string_length
)
from ..utils.real_world import (
    ErrorCode,
    validate_pdf_file,
    format_error_response,
    retry_with_backoff,
    RetryStrategy,
    with_timeout
)
from ..utils.resilience import with_fallback, degrade_gracefully
from ..utils.response_helpers import get_request_id, create_error_response, create_success_response
from ..utils.structured_logging import set_request_context
from .dependencies import get_services

logger = logging.getLogger(__name__)

# PDF Processing Router
pdf_router = APIRouter()

@pdf_router.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    auto_process: bool = Form(True),
    extract_text: bool = Form(True),
    detect_language: bool = Form(True),
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload and process a PDF document with real-world error handling"""
    try:
        # Enforce size and type limits (defensive)
        allowed_types = {"application/pdf"}
        max_size_mb = 100
        max_size_bytes = max_size_mb * 1024 * 1024

        # Real-world validation with specific error codes
        request_id = get_request_id(request)
        
        # Set request context for logging
        set_request_context(request_id=request_id, user_id=user_id)
        
        # Validate filename
        if file.filename:
            is_valid, error_msg = validate_filename(file.filename)
            if not is_valid:
                error_response = format_error_response(
                    ErrorCode.VALIDATION_ERROR,
                    f"Invalid filename: {error_msg}",
                    {"filename": file.filename},
                    request_id
                )
                raise HTTPException(status_code=400, detail=error_response)
            
            # Validate file extension
            is_valid, error_msg = validate_file_extension(file.filename, [".pdf"])
            if not is_valid:
                error_response = format_error_response(
                    ErrorCode.PDF_INVALID_FORMAT,
                    error_msg or "Invalid file extension",
                    {"filename": file.filename},
                    request_id
                )
                raise HTTPException(status_code=400, detail=error_response)
        
        if file.content_type not in allowed_types:
            error_response = format_error_response(
                ErrorCode.PDF_INVALID_FORMAT,
                "Unsupported media type. Only application/pdf files are allowed.",
                {"received_type": file.content_type},
                request_id
            )
            raise HTTPException(status_code=415, detail=error_response)

        # Check file size
        content_length = 0
        try:
            content_length = int(file.headers.get("content-length", "0")) if hasattr(file, "headers") else 0
        except Exception:
            content_length = 0
        
        if content_length and content_length > max_size_bytes:
            error_response = format_error_response(
                ErrorCode.PDF_TOO_LARGE,
                f"PDF file exceeds maximum size of {max_size_mb}MB.",
                {
                    "max_size_mb": max_size_mb,
                    "file_size_mb": round(content_length / (1024 * 1024), 2)
                },
                request_id
            )
            raise HTTPException(status_code=413, detail=error_response)

        # Read and validate PDF file content
        file_content = await file.read()
        
        # Real-world PDF validation
        is_valid, error_msg = validate_pdf_file(file_content, max_size_mb)
        if not is_valid:
            error_code = ErrorCode.PDF_CORRUPTED
            if "encrypted" in error_msg.lower():
                error_code = ErrorCode.PDF_ENCRYPTED
            elif "invalid" in error_msg.lower():
                error_code = ErrorCode.PDF_INVALID_FORMAT
            
            error_response = format_error_response(
                error_code,
                error_msg or "PDF file validation failed.",
                {"filename": file.filename},
                request_id
            )
            raise HTTPException(status_code=400, detail=error_response)
        
        # Reset file pointer for service to read
        await file.seek(0)
        
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            error_response = format_error_response(
                ErrorCode.SERVICE_UNAVAILABLE,
                "PDF processing service is currently unavailable. Please try again later.",
                None,
                request_id
            )
            raise HTTPException(status_code=503, detail=error_response)
        
        # Create upload request
        upload_request = PDFUploadRequest(
            filename=file.filename,
            auto_process=auto_process,
            extract_text=extract_text,
            detect_language=detect_language
        )
        
        # Process upload with timeout and retry
        @with_timeout(timeout_seconds=300.0)  # 5 minute timeout for large files
        @retry_with_backoff(max_attempts=2, initial_delay=1.0)
        async def process_upload():
            return await pdf_service.upload_pdf(file, upload_request, user_id)
        
        try:
            result = await process_upload()
        except asyncio.TimeoutError:
            error_response = format_error_response(
                ErrorCode.SERVICE_TIMEOUT,
                "PDF processing timed out. The file may be too large or complex. Please try a smaller file.",
                {"filename": file.filename},
                request_id
            )
            raise HTTPException(status_code=504, detail=error_response)
        except Exception as e:
            logger.error(f"PDF upload processing failed: {e}", exc_info=True)
            error_response = format_error_response(
                ErrorCode.PDF_PROCESSING_FAILED,
                f"Failed to process PDF file: {str(e)}",
                {"filename": file.filename},
                request_id
            )
            raise HTTPException(status_code=500, detail=error_response)
        
        # Track analytics
        if services.get("analytics_service"):
            background_tasks.add_task(
                services["analytics_service"].track_event,
                "pdf_uploaded",
                user_id,
                {"document_id": result.document.id, "user_id": user_id}
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        request_id = get_request_id(request)
        logger.error(f"[{request_id}] Error uploading PDF: {e}", exc_info=True)
        error_response = format_error_response(
            ErrorCode.INTERNAL_ERROR,
            "An unexpected error occurred while processing your PDF. Please try again.",
            None,
            request_id
        )
        raise HTTPException(status_code=500, detail=error_response)

@pdf_router.get(
    "/documents",
    response_model=List[PDFDocument],
    summary="List PDF documents",
    description="Get a list of PDF documents for the current user"
)
async def list_documents(
    request: Request,
    user_id: str = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100, description="Number of items per page"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    services: Dict[str, Any] = Depends(get_services)
):
    """List user's PDF documents with improved response and validation"""
    request_id = get_request_id(request)
    set_request_context(request_id=request_id, user_id=user_id)
    
    try:
        # Validate pagination parameters
        is_valid, error_msg = validate_integer_range(limit, 1, 100, "limit")
        if not is_valid:
            error_response = format_error_response(
                ErrorCode.VALIDATION_ERROR,
                error_msg or "Invalid limit parameter",
                {"limit": limit},
                request_id
            )
            raise HTTPException(status_code=400, detail=error_response)
        
        is_valid, error_msg = validate_integer_range(offset, 0, None, "offset")
        if not is_valid:
            error_response = format_error_response(
                ErrorCode.VALIDATION_ERROR,
                error_msg or "Invalid offset parameter",
                {"offset": offset},
                request_id
            )
            raise HTTPException(status_code=400, detail=error_response)
        
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            error_response = format_error_response(
                ErrorCode.SERVICE_UNAVAILABLE,
                "PDF service is currently unavailable",
                None,
                request_id
            )
            raise HTTPException(status_code=503, detail=error_response)
        
        documents = await pdf_service.list_documents(user_id, limit, offset)
        
        # Create success response with metadata
        response_data = create_success_response(
            data=documents,
            message=f"Retrieved {len(documents)} documents",
            metadata={
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "count": len(documents)
                }
            },
            request_id=request_id
        )
        
        # Add headers for caching and pagination
        response = JSONResponse(content=response_data)
        response.headers["X-Total-Count"] = str(len(documents))
        response.headers["Cache-Control"] = "public, max-age=60"
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@pdf_router.get("/documents/{document_id}", response_model=PDFDocument)
async def get_document(
    request: Request,
    document_id: str = Path(..., description="Document ID"),
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
):
    """Get a specific PDF document with enhanced validation"""
    request_id = get_request_id(request)
    set_request_context(request_id=request_id, user_id=user_id)
    
    try:
        # Validate document ID format
        is_valid, error_msg = Validator.validate_document_id(document_id)
        if not is_valid:
            error_response = format_error_response(
                ErrorCode.VALIDATION_ERROR,
                error_msg or "Invalid document ID format",
                {"document_id": document_id},
                request_id
            )
            raise HTTPException(status_code=400, detail=error_response)
        
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            error_response = format_error_response(
                ErrorCode.SERVICE_UNAVAILABLE,
                "PDF service is currently unavailable",
                None,
                request_id
            )
            raise HTTPException(status_code=503, detail=error_response)
        
        document = await pdf_service.get_document(document_id, user_id)
        if not document:
            error_response = format_error_response(
                ErrorCode.NOT_FOUND,
                f"Document not found: {document_id}",
                {"document_id": document_id},
                request_id
            )
            raise HTTPException(status_code=404, detail=error_response)
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}", exc_info=True, extra={"document_id": document_id})
        error_response = format_error_response(
            ErrorCode.INTERNAL_ERROR,
            "An error occurred while retrieving the document",
            None,
            request_id
        )
        raise HTTPException(status_code=500, detail=error_response)

@pdf_router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str = Path(..., description="Document ID"),
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
):
    """Delete a PDF document"""
    try:
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            raise HTTPException(status_code=500, detail="PDF service not available")
        
        success = await pdf_service.delete_document(document_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Variant Generation Router
variant_router = APIRouter()

@variant_router.post("/generate", response_model=VariantGenerateResponse)
async def generate_variants(
    request: VariantGenerateRequest,
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Generate variants of a PDF document"""
    try:
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            raise HTTPException(status_code=500, detail="PDF service not available")
        
        # Check document access
        document = await pdf_service.get_document(request.document_id, user_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Generate variants
        result = await pdf_service.generate_variants(request, user_id)
        
        # Track analytics
        if services.get("analytics_service"):
            background_tasks.add_task(
                services["analytics_service"].track_event,
                "variants_generated",
                user_id,
                {
                    "document_id": request.document_id,
                    "user_id": user_id,
                    "count": result.total_generated
                }
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating variants: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@variant_router.get("/documents/{document_id}/variants", response_model=List[PDFVariant])
async def list_variants(
    document_id: str = Path(..., description="Document ID"),
    user_id: str = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    services: Dict[str, Any] = Depends(get_services)
):
    """List variants for a document"""
    try:
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            raise HTTPException(status_code=500, detail="PDF service not available")
        
        # Check document access
        document = await pdf_service.get_document(document_id, user_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        variants = await pdf_service.list_variants(document_id, user_id, limit, offset)
        return variants
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing variants: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@variant_router.get("/variants/{variant_id}", response_model=PDFVariant)
async def get_variant(
    variant_id: str = Path(..., description="Variant ID"),
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
):
    """Get a specific variant"""
    try:
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            raise HTTPException(status_code=500, detail="PDF service not available")
        
        variant = await pdf_service.get_variant(variant_id, user_id)
        if not variant:
            raise HTTPException(status_code=404, detail="Variant not found")
        
        return variant
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting variant: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@variant_router.post("/stop")
async def stop_generation(
    document_id: str = Form(...),
    keep_generated: bool = Form(True),
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
):
    """Stop variant generation for a document"""
    try:
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            raise HTTPException(status_code=500, detail="PDF service not available")
        
        result = await pdf_service.stop_generation(document_id, keep_generated, user_id)
        return result
        
    except Exception as e:
        logger.error(f"Error stopping generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Topic Extraction Router
topic_router = APIRouter()

@topic_router.post("/extract", response_model=TopicExtractResponse)
async def extract_topics(
    request: TopicExtractRequest,
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Extract topics from a PDF document"""
    try:
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            raise HTTPException(status_code=500, detail="PDF service not available")
        
        # Check document access
        document = await pdf_service.get_document(request.document_id, user_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Extract topics
        result = await pdf_service.extract_topics(request, user_id)
        
        # Track analytics
        if services.get("analytics_service"):
            background_tasks.add_task(
                services["analytics_service"].track_event,
                "topics_extracted",
                user_id,
                {
                    "document_id": request.document_id,
                    "user_id": user_id,
                    "count": result.total_topics
                }
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@topic_router.get("/documents/{document_id}/topics", response_model=List[TopicItem])
async def list_topics(
    document_id: str = Path(..., description="Document ID"),
    user_id: str = Depends(get_current_user),
    min_relevance: float = Query(0.5, ge=0.0, le=1.0),
    services: Dict[str, Any] = Depends(get_services)
):
    """List topics for a document"""
    try:
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            raise HTTPException(status_code=500, detail="PDF service not available")
        
        # Check document access
        document = await pdf_service.get_document(document_id, user_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        topics = await pdf_service.list_topics(document_id, user_id, min_relevance)
        return topics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Brainstorming Router
brainstorm_router = APIRouter()

@brainstorm_router.post("/generate", response_model=BrainstormGenerateResponse)
async def generate_brainstorm_ideas(
    request: BrainstormGenerateRequest,
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Generate brainstorm ideas from a PDF document"""
    try:
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            raise HTTPException(status_code=500, detail="PDF service not available")
        
        # Check document access
        document = await pdf_service.get_document(request.document_id, user_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Generate brainstorm ideas
        result = await pdf_service.generate_brainstorm_ideas(request, user_id)
        
        # Track analytics
        if services.get("analytics_service"):
            background_tasks.add_task(
                services["analytics_service"].track_event,
                "brainstorm_generated",
                user_id,
                {
                    "document_id": request.document_id,
                    "user_id": user_id,
                    "count": result.total_ideas
                }
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating brainstorm ideas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@brainstorm_router.get("/documents/{document_id}/ideas", response_model=List[BrainstormIdea])
async def list_brainstorm_ideas(
    document_id: str = Path(..., description="Document ID"),
    user_id: str = Depends(get_current_user),
    category: Optional[str] = Query(None),
    services: Dict[str, Any] = Depends(get_services)
):
    """List brainstorm ideas for a document"""
    try:
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            raise HTTPException(status_code=500, detail="PDF service not available")
        
        # Check document access
        document = await pdf_service.get_document(document_id, user_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        ideas = await pdf_service.list_brainstorm_ideas(document_id, user_id, category)
        return ideas
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing brainstorm ideas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Collaboration Router
collaboration_router = APIRouter()

@collaboration_router.post("/invite")
async def invite_collaborator(
    invite: CollaborationInvite,
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
):
    """Invite a collaborator to a document"""
    try:
        collaboration_service: CollaborationService = services.get("collaboration_service")
        if not collaboration_service:
            raise HTTPException(status_code=500, detail="Collaboration service not available")
        
        result = await collaboration_service.invite_collaborator(invite, user_id)
        return result
        
    except Exception as e:
        logger.error(f"Error inviting collaborator: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@collaboration_router.websocket("/ws/{document_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    document_id: str = Path(..., description="Document ID"),
    user_id: Optional[str] = Query(None, description="User ID"),
):
    """WebSocket endpoint for real-time collaboration"""
    try:
        # Get services from dependencies
        from .dependencies import get_services
        global_services = get_services()
        
        # Get user_id from query or default to anonymous in dev mode
        if not user_id:
            import os
            dev_mode = os.getenv("ENVIRONMENT", "development").lower() == "development"
            user_id = "anonymous" if dev_mode else None
        
        if not user_id:
            await websocket.close(code=1008, reason="User ID required")
            return
        
        collaboration_service: CollaborationService = global_services.get("collaboration_service")
        if not collaboration_service:
            await websocket.close(code=1011, reason="Collaboration service not available")
            return
        
        await collaboration_service.connect_user(websocket, document_id, user_id)
        
        try:
            while True:
                data = await websocket.receive_text()
                await collaboration_service.handle_message(document_id, user_id, data)
        except WebSocketDisconnect:
            await collaboration_service.disconnect_user(document_id, user_id)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason="Internal server error")

# Export Router
export_router = APIRouter()

@export_router.post("/export", response_model=ExportResponse)
async def export_content(
    request: ExportRequest,
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Export content to specified format"""
    try:
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            raise HTTPException(status_code=500, detail="PDF service not available")
        
        # Check document access
        document = await pdf_service.get_document(request.document_id, user_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Export content
        result = await pdf_service.export_content(request, user_id)
        
        # Track analytics
        if services.get("analytics_service"):
            background_tasks.add_task(
                services["analytics_service"].track_event,
                "content_exported",
                user_id,
                {
                    "document_id": request.document_id,
                    "user_id": user_id,
                    "format": request.export_format
                }
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@export_router.get("/download/{file_id}")
async def download_file(
    file_id: str = Path(..., description="File ID"),
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
):
    """Download exported file"""
    try:
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            raise HTTPException(status_code=500, detail="PDF service not available")
        
        file_info = await pdf_service.get_exported_file(file_id, user_id)
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            file_info["path"],
            filename=file_info["filename"],
            media_type=file_info["media_type"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics Router
analytics_router = APIRouter()

@analytics_router.get("/dashboard")
async def get_dashboard_data(
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
):
    """Get dashboard analytics data"""
    try:
        analytics_service: AnalyticsService = services.get("analytics_service")
        if not analytics_service:
            raise HTTPException(status_code=500, detail="Analytics service not available")
        
        dashboard_data = await analytics_service.get_dashboard_data(user_id)
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@analytics_router.get("/reports")
async def get_analytics_report(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
):
    """Get analytics report for date range"""
    try:
        analytics_service: AnalyticsService = services.get("analytics_service")
        if not analytics_service:
            raise HTTPException(status_code=500, detail="Analytics service not available")
        
        report = await analytics_service.generate_report(start_date, end_date, user_id)
        return report
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search Router
search_router = APIRouter()

@search_router.post("/search", response_model=SearchResponse)
async def search_content(
    request: SearchRequest,
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services)
):
    """Search across documents and variants"""
    try:
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            raise HTTPException(status_code=500, detail="PDF service not available")
        
        results = await pdf_service.search_content(request, user_id)
        return results
        
    except Exception as e:
        logger.error(f"Error searching content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch Processing Router
batch_router = APIRouter()

@batch_router.post("/process", response_model=BatchProcessingResponse)
async def batch_process(
    request: BatchProcessingRequest,
    user_id: str = Depends(get_current_user),
    services: Dict[str, Any] = Depends(get_services),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Process multiple documents in batch"""
    try:
        pdf_service: PDFVariantesService = services.get("pdf_service")
        if not pdf_service:
            raise HTTPException(status_code=500, detail="PDF service not available")
        
        # Process batch
        result = await pdf_service.batch_process(request, user_id)
        
        # Track analytics
        if services.get("analytics_service"):
            background_tasks.add_task(
                services["analytics_service"].track_event,
                "batch_processed",
                user_id,
                {
                    "user_id": user_id,
                    "document_count": len(request.document_ids) if hasattr(request, 'document_ids') else 0,
                    "operation": request.operation
                }
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health Router
health_router = APIRouter()

@health_router.get("/status")
async def get_system_health(
    request: Request,
    services: Dict[str, Any] = Depends(get_services)
):
    """Get real-world system health status with dependency checks"""
    request_id = get_request_id(request) if request else None
    
    try:
        # Use real-world health check if available
        health_check = services.get("health_check")
        if health_check:
            health_status = await health_check.check_all()
            
            # Map to SystemHealth format if needed
            return {
                "status": health_status["status"],
                "timestamp": health_status["timestamp"],
                "checks": health_status["checks"],
                "request_id": request_id
            }
        
        # Fallback to health service
        health_service = services.get("health_service")
        if health_service:
            health_status = await health_service.get_system_health()
            return health_status
        
        # Default health response
        return {
            "status": "unknown",
            "message": "Health check service not available",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Error getting system health: {e}", exc_info=True)
        error_response = format_error_response(
            ErrorCode.SERVICE_UNAVAILABLE,
            "Unable to check system health",
            None,
            request_id
        )
        raise HTTPException(status_code=503, detail=error_response)
