"""Document routes"""

from fastapi import APIRouter, UploadFile, File
from typing import List, Optional

from models.schemas import (
    DocumentRequest,
    DocumentResponse,
)
from utils.dependencies import DocumentServiceDep
from utils.exceptions import NotFoundError, ValidationError
from config.settings import settings

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", response_model=DocumentResponse, status_code=201)
async def upload_document(
    shipment_id: str,
    document_type: str,
    file: UploadFile = File(...),
    description: Optional[str] = None,
    document_service: DocumentServiceDep = None
) -> DocumentResponse:
    """Upload a document"""
    # Validate file size
    file_content = await file.read()
    if len(file_content) > settings.MAX_UPLOAD_SIZE:
        raise ValidationError(
            f"File size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE} bytes",
            field="file"
        )
    
    mime_type = file.content_type or "application/octet-stream"
    
    request = DocumentRequest(
        shipment_id=shipment_id,
        document_type=document_type,
        file_name=file.filename or "unknown",
        description=description
    )
    
    document = await document_service.upload_document(
        request=request,
        file_content=file_content,
        mime_type=mime_type
    )
    return document


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    document_service: DocumentServiceDep
) -> DocumentResponse:
    """Get document by ID"""
    document = await document_service.get_document(document_id)
    if not document:
        raise NotFoundError("Document", document_id)
    return document


@router.get("/shipment/{shipment_id}", response_model=List[DocumentResponse])
async def get_documents_by_shipment(
    shipment_id: str,
    document_service: DocumentServiceDep
) -> List[DocumentResponse]:
    """Get documents for a shipment"""
    documents = await document_service.get_documents_by_shipment(shipment_id)
    return documents


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    document_service: DocumentServiceDep
):
    """Delete a document"""
    success = await document_service.delete_document(document_id)
    if not success:
        raise NotFoundError("Document", document_id)
    return {"message": "Document deleted successfully"}

