"""
Documents API Router
====================

API endpoints for document operations.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List, Optional
import logging

from ..document_generator import DocumentType, DocumentFormat
from ..schemas.document_schemas import (
    DocumentRequestModel, DocumentResponse, DocumentListResponse,
    DocumentGenerationResponse, DocumentTemplateResponse
)
from ..core.dependencies import get_document_service
from ..core.exceptions import convert_to_http_exception
from ..services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/generate", response_model=DocumentGenerationResponse)
async def generate_document(
    request: DocumentRequestModel,
    created_by: str = Query(..., description="User creating the document"),
    document_service: DocumentService = Depends(get_document_service)
):
    """Generate a business document."""
    
    try:
        result = await document_service.generate_document(
            document_type=request.document_type,
            title=request.title,
            description=request.description,
            business_area=request.business_area,
            created_by=created_by,
            variables=request.variables,
            format=request.format
        )
        
        return DocumentGenerationResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to generate document: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate document")

@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    business_area: Optional[str] = Query(None, description="Filter by business area"),
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    created_by: Optional[str] = Query(None, description="Filter by creator"),
    document_service: DocumentService = Depends(get_document_service)
):
    """List generated documents with optional filters."""
    
    try:
        documents_data = await document_service.list_documents(
            business_area=business_area,
            document_type=document_type,
            created_by=created_by
        )
        
        return DocumentListResponse(
            documents=documents_data,
            total=len(documents_data),
            business_area=business_area,
            document_type=document_type.value if document_type else None
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list documents")

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service)
):
    """Get specific document details."""
    
    try:
        document_data = await document_service.get_document(document_id)
        return DocumentResponse(**document_data)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get document")

@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service)
):
    """Download a generated document."""
    
    try:
        file_path = await document_service.get_document_file_path(document_id)
        
        # Get document info for filename
        document_data = await document_service.get_document(document_id)
        
        return FileResponse(
            path=file_path,
            filename=f"{document_data['title']}.{document_data['format']}",
            media_type="application/octet-stream"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to download document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download document")
