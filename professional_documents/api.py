"""
Professional Documents API
==========================

FastAPI router for professional document generation and management endpoints.
"""

import logging
from typing import List, Optional
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
from .services import DocumentGenerationService, DocumentExportService, TemplateService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/professional-documents", tags=["Professional Documents"])

# Initialize services
document_service = DocumentGenerationService()
export_service = DocumentExportService()
template_service = TemplateService()


@router.post("/generate", response_model=DocumentGenerationResponse)
async def generate_document(
    request: DocumentGenerationRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Generate a professional document based on user query.
    
    This endpoint creates a professional document using AI-powered content generation
    with customizable templates, styling, and formatting options.
    """
    try:
        logger.info(f"Generating document for user {user.id}: {request.document_type}")
        
        # Generate document
        response = await document_service.generate_document(request)
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)
        
        logger.info(f"Document generated successfully: {response.document.id}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate document: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Number of documents per page"),
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    List user's documents with pagination and filtering.
    """
    try:
        # Get documents with pagination
        offset = (page - 1) * page_size
        documents = document_service.list_documents(limit=page_size, offset=offset)
        
        # Filter by document type if specified
        if document_type:
            documents = [doc for doc in documents if doc.document_type == document_type]
        
        # Calculate total count (simplified - in production, this would be a proper count query)
        total_count = len(document_service.list_documents(limit=1000, offset=0))
        
        return DocumentListResponse(
            documents=documents,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/documents/{document_id}", response_model=ProfessionalDocument)
async def get_document(
    document_id: str,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Get a specific document by ID.
    """
    try:
        document = document_service.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@router.put("/documents/{document_id}", response_model=ProfessionalDocument)
async def update_document(
    document_id: str,
    request: DocumentUpdateRequest,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Update an existing document.
    """
    try:
        # Convert request to dict, excluding None values
        updates = {}
        if request.title is not None:
            updates["title"] = request.title
        if request.subtitle is not None:
            updates["subtitle"] = request.subtitle
        if request.sections is not None:
            updates["sections"] = request.sections
        if request.style is not None:
            updates["style"] = request.style
        if request.metadata is not None:
            updates["metadata"] = request.metadata
        
        document = document_service.update_document(document_id, updates)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update document: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Delete a document.
    """
    try:
        document = document_service.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove from service (in production, this would be a proper database delete)
        if document_id in document_service.documents:
            del document_service.documents[document_id]
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.post("/export", response_model=DocumentExportResponse)
async def export_document(
    request: DocumentExportRequest,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Export a document in the specified format (PDF, MD, Word, HTML).
    """
    try:
        # Get the document
        document = document_service.get_document(request.document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Export the document
        response = await export_service.export_document(document, request)
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export document: {str(e)}")


@router.get("/download/{filename}")
async def download_file(
    filename: str,
    user: User = Depends(current_user)
):
    """
    Download an exported file.
    """
    try:
        # Security check - ensure filename is safe
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = export_service.output_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")


@router.get("/templates", response_model=TemplateListResponse)
async def list_templates(
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    user: User = Depends(current_user)
):
    """
    List available document templates.
    """
    try:
        if document_type:
            templates = template_service.get_templates_by_type(document_type)
        else:
            templates = template_service.get_all_templates()
        
        return TemplateListResponse(
            templates=templates,
            total_count=len(templates)
        )
        
    except Exception as e:
        logger.error(f"Error listing templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")


@router.get("/templates/{template_id}")
async def get_template(
    template_id: str,
    user: User = Depends(current_user)
):
    """
    Get a specific template by ID.
    """
    try:
        template = template_service.get_template(template_id)
        return template
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting template {template_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")


@router.get("/stats", response_model=DocumentStats)
async def get_document_stats(
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Get document statistics for the user.
    """
    try:
        # Get all documents
        all_documents = document_service.list_documents(limit=1000, offset=0)
        
        # Calculate stats
        total_documents = len(all_documents)
        documents_by_type = {}
        total_word_count = 0
        
        for doc in all_documents:
            # Count by type
            doc_type = doc.document_type.value
            documents_by_type[doc_type] = documents_by_type.get(doc_type, 0) + 1
            
            # Sum word count
            total_word_count += doc.word_count
        
        average_document_length = total_word_count / total_documents if total_documents > 0 else 0
        
        # Most used templates (simplified)
        most_used_templates = [
            {"template_id": "template1", "name": "Business Report", "usage_count": 5},
            {"template_id": "template2", "name": "Proposal", "usage_count": 3}
        ]
        
        # Export stats (simplified)
        export_stats = {
            "pdf": 10,
            "docx": 5,
            "md": 3,
            "html": 2
        }
        
        return DocumentStats(
            total_documents=total_documents,
            documents_by_type=documents_by_type,
            total_word_count=total_word_count,
            average_document_length=average_document_length,
            most_used_templates=most_used_templates,
            export_stats=export_stats
        )
        
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document stats: {str(e)}")


@router.post("/documents/{document_id}/regenerate")
async def regenerate_document(
    document_id: str,
    request: DocumentGenerationRequest,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session)
):
    """
    Regenerate an existing document with new content.
    """
    try:
        # Check if document exists
        existing_document = document_service.get_document(document_id)
        
        if not existing_document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Generate new document
        response = await document_service.generate_document(request)
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.message)
        
        # Update the existing document with new content
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to regenerate document: {str(e)}")


@router.get("/formats")
async def get_supported_formats():
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
async def health_check():
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




























