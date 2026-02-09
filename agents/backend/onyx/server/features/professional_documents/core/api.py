"""
Professional Documents API
==========================

FastAPI router for professional document generation and management endpoints.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.db.engine import get_session
from onyx.db.models import User

from .models import (
    DocumentGenerationRequest,
    DocumentGenerationResponse,
    DocumentExportRequest,
    DocumentExportResponse,
    DocumentListResponse,
    DocumentUpdateRequest,
    TemplateListResponse,
    DocumentStats,
    DocumentType,
    ExportFormat,
    ProfessionalDocument
)
from .utils import handle_api_errors
from .validators import validate_document_exists, validate_export_request, validate_filename
from .dependencies import (
    get_document_generation_service,
    get_document_export_service,
    get_template_service
)
from .services import DocumentGenerationService, DocumentExportService, TemplateService
from .stats import calculate_document_stats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/professional-documents", tags=["Professional Documents"])


@router.post("/generate", response_model=DocumentGenerationResponse)
@handle_api_errors
async def generate_document(
    request: DocumentGenerationRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
    document_service: DocumentGenerationService = Depends(get_document_generation_service)
):
    """
    Generate a professional document based on user query.
    
    This endpoint creates a professional document using AI-powered content generation
    with customizable templates, styling, and formatting options.
    """
    logger.info(f"Generating document for user {user.id}: {request.document_type}")
    
    response = await document_service.generate_document(request)
    
    if not response.success:
        raise HTTPException(status_code=500, detail=response.message)
    
    logger.info(f"Document generated successfully: {response.document.id}")
    return response


@router.get("/documents", response_model=DocumentListResponse)
@handle_api_errors
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Number of documents per page"),
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
    document_service: DocumentGenerationService = Depends(get_document_generation_service)
):
    """
    List user's documents with pagination and filtering.
    """
    offset = (page - 1) * page_size
    
    # Get total count efficiently
    total_count = document_service.count_documents(document_type=document_type)
    
    # Get paginated results
    documents = document_service.list_documents(limit=page_size, offset=offset, document_type=document_type)
    
    return DocumentListResponse(
        documents=documents,
        total_count=total_count,
        page=page,
        page_size=page_size
    )


@router.get("/documents/{document_id}", response_model=ProfessionalDocument)
@handle_api_errors
async def get_document(
    document_id: str,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
    document_service: DocumentGenerationService = Depends(get_document_generation_service)
):
    """
    Get a specific document by ID.
    """
    document = document_service.get_document(document_id)
    validate_document_exists(document, document_id)
    return document


@router.put("/documents/{document_id}", response_model=ProfessionalDocument)
@handle_api_errors
async def update_document(
    document_id: str,
    request: DocumentUpdateRequest,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
    document_service: DocumentGenerationService = Depends(get_document_generation_service)
):
    """
    Update an existing document.
    """
    updates = {
        k: v for k, v in {
            "title": request.title,
            "subtitle": request.subtitle,
            "sections": request.sections,
            "style": request.style,
            "metadata": request.metadata
        }.items() if v is not None
    }
    
    document = document_service.update_document(document_id, updates)
    return document


@router.delete("/documents/{document_id}")
@handle_api_errors
async def delete_document(
    document_id: str,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
    document_service: DocumentGenerationService = Depends(get_document_generation_service)
):
    """
    Delete a document.
    """
    document = document_service.get_document(document_id)
    validate_document_exists(document, document_id)
    document_service.storage.delete(document_id)
    
    return {"message": "Document deleted successfully"}


@router.post("/export", response_model=DocumentExportResponse)
@handle_api_errors
async def export_document(
    request: DocumentExportRequest,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
    document_service: DocumentGenerationService = Depends(get_document_generation_service),
    export_service: DocumentExportService = Depends(get_document_export_service)
):
    """
    Export a document in the specified format (PDF, MD, Word, HTML).
    """
    document = document_service.get_document(request.document_id)
    validate_document_exists(document, request.document_id)
    validate_export_request(request)
    
    response = await export_service.export_document(document, request)
    
    if not response.success:
        raise HTTPException(status_code=500, detail=response.message)
    
    return response


@router.get("/download/{filename}")
@handle_api_errors
async def download_file(
    filename: str,
    user: User = Depends(current_user),
    export_service: DocumentExportService = Depends(get_document_export_service)
):
    """
    Download an exported file.
    """
    validate_filename(filename)
    file_path = export_service.output_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/octet-stream'
    )


@router.get("/templates", response_model=TemplateListResponse)
@handle_api_errors
async def list_templates(
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    user: User = Depends(current_user),
    template_service: TemplateService = Depends(get_template_service)
):
    """
    List available document templates.
    """
    if document_type:
        templates = template_service.get_templates_by_type(document_type)
    else:
        templates = template_service.get_all_templates()
    
    return TemplateListResponse(
        templates=templates,
        total_count=len(templates)
    )


@router.get("/templates/{template_id}")
@handle_api_errors
async def get_template(
    template_id: str,
    user: User = Depends(current_user),
    template_service: TemplateService = Depends(get_template_service)
):
    """
    Get a specific template by ID.
    """
    try:
        template = template_service.get_template(template_id)
        return template
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/stats", response_model=DocumentStats)
@handle_api_errors
async def get_document_stats(
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
    document_service: DocumentGenerationService = Depends(get_document_generation_service),
    template_service: TemplateService = Depends(get_template_service)
):
    """
    Get document statistics for the user.
    """
    all_documents = document_service.list_documents(limit=1000, offset=0)
    stats = calculate_document_stats(all_documents)
    
    # TODO: Implement actual template usage tracking
    most_used_templates = []
    
    # TODO: Implement actual export statistics tracking
    export_stats = {
        "pdf": 0,
        "docx": 0,
        "md": 0,
        "html": 0
    }
    
    return DocumentStats(
        total_documents=stats["total_documents"],
        documents_by_type=stats["documents_by_type"],
        total_word_count=stats["total_word_count"],
        average_document_length=stats["average_document_length"],
        most_used_templates=most_used_templates,
        export_stats=export_stats
    )


@router.post("/documents/{document_id}/regenerate")
@handle_api_errors
async def regenerate_document(
    document_id: str,
    request: DocumentGenerationRequest,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
    document_service: DocumentGenerationService = Depends(get_document_generation_service)
):
    """
    Regenerate an existing document with new content.
    """
    existing_document = document_service.get_document(document_id)
    validate_document_exists(existing_document, document_id)
    
    response = await document_service.generate_document(request)
    
    if not response.success:
        raise HTTPException(status_code=500, detail=response.message)
    
    updates = {
        "sections": response.document.sections,
        "word_count": response.document.word_count,
        "page_count": response.document.page_count,
        "status": "completed"
    }
    
    updated_document = document_service.update_document(document_id, updates)
    
    return DocumentGenerationResponse(
        success=True,
        document=updated_document,
        message="Document regenerated successfully",
        generation_time=response.generation_time,
        word_count=response.word_count,
        estimated_pages=response.estimated_pages
    )


@router.get("/formats")
async def get_supported_formats() -> Dict[str, Any]:
    """
    Get list of supported export formats.
    """
    return {
        "formats": [
            {
                "format": "pdf",
                "name": "PDF Document",
                "description": "Portable Document Format with professional formatting",
                "mime_type": "application/pdf"
            },
            {
                "format": "docx",
                "name": "Microsoft Word",
                "description": "Word document with full formatting support",
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            },
            {
                "format": "md",
                "name": "Markdown",
                "description": "Markdown format for easy editing and version control",
                "mime_type": "text/markdown"
            },
            {
                "format": "html",
                "name": "HTML Document",
                "description": "Web-ready HTML document with embedded styles",
                "mime_type": "text/html"
            }
        ]
    }


@router.get("/health")
async def health_check(
    template_service: TemplateService = Depends(get_template_service)
) -> Dict[str, Any]:
    """
    Health check endpoint for the professional documents service.
    """
    return {
        "status": "healthy",
        "service": "professional-documents",
        "version": "1.0.0",
        "features": {
            "document_generation": True,
            "export_formats": ["pdf", "docx", "md", "html"],
            "templates": len(template_service.get_all_templates()),
            "ai_generation": True
        }
    }

