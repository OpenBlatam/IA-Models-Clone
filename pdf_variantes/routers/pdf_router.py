"""Core PDF router with functional approach."""

from fastapi import APIRouter, UploadFile, File, Depends, Query, Path, HTTPException
from typing import Dict, Any, List, Optional
import logging

from ..utils import validate_pdf_file, extract_metadata, sanitize_filename
from ..dependencies import get_pdf_service, get_current_user, validate_file_size
from ..exceptions import PDFNotFoundError, InvalidFileError
from ..schemas import PDFUploadSchema, VariantGenerateSchema

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pdf", tags=["PDF Processing"])


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    auto_process: bool = Query(True),
    extract_text: bool = Query(True),
    pdf_service = Depends(get_pdf_service),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Upload PDF with validation and processing."""
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise InvalidFileError("Only PDF files are allowed")
    
    file_content = await file.read()
    validate_file_size(len(file_content))
    
    # Validate PDF
    validation = validate_pdf_file(file_content, file.filename)
    if not validation["valid"]:
        raise InvalidFileError(validation["error"])
    
    # Extract metadata
    metadata = extract_metadata(file_content)
    sanitized_filename = sanitize_filename(file.filename)
    
    # Process upload
    pdf_metadata, text_content = await pdf_service.upload_handler.upload_pdf(
        file_content=file_content,
        filename=sanitized_filename,
        auto_process=auto_process,
        extract_text=extract_text
    )
    
    return {
        "success": True,
        "file_id": pdf_metadata.file_id,
        "metadata": metadata,
        "processing_started": auto_process
    }


@router.get("/{file_id}/preview")
async def get_preview(
    file_id: str = Path(...),
    page_number: int = Query(1, ge=1),
    pdf_service = Depends(get_pdf_service)
) -> Dict[str, Any]:
    """Get PDF page preview."""
    preview = await pdf_service.upload_handler.get_pdf_preview(file_id, page_number)
    
    if not preview:
        raise PDFNotFoundError(f"PDF {file_id} not found")
    
    return {
        "file_id": file_id,
        "page_number": page_number,
        "preview": preview
    }


@router.post("/{file_id}/variants")
async def generate_variant(
    file_id: str = Path(...),
    variant_request: VariantGenerateSchema = ...,
    pdf_service = Depends(get_pdf_service)
) -> Dict[str, Any]:
    """Generate PDF variant."""
    file_path = pdf_service.upload_handler.get_file_path(file_id)
    
    if not file_path.exists():
        raise PDFNotFoundError(f"PDF {file_id} not found")
    
    with open(file_path, "rb") as f:
        variant = await pdf_service.variant_generator.generate(
            file=f,
            variant_type=variant_request.variant_type,
            options=variant_request.options
        )
    
    return {
        "success": True,
        "variant": variant,
        "file_id": file_id
    }


@router.get("/{file_id}/topics")
async def extract_topics(
    file_id: str = Path(...),
    min_relevance: float = Query(0.5, ge=0.0, le=1.0),
    max_topics: int = Query(50, ge=1, le=200),
    pdf_service = Depends(get_pdf_service)
) -> Dict[str, Any]:
    """Extract topics from PDF."""
    topics = await pdf_service.topic_extractor.extract_topics(
        file_id=file_id,
        min_relevance=min_relevance,
        max_topics=max_topics
    )
    
    return {
        "file_id": file_id,
        "topics": [topic.to_dict() if hasattr(topic, 'to_dict') else topic for topic in topics],
        "total_count": len(topics)
    }


@router.delete("/{file_id}")
async def delete_pdf(
    file_id: str = Path(...),
    pdf_service = Depends(get_pdf_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, bool]:
    """Delete PDF file."""
    success = await pdf_service.upload_handler.delete_pdf(file_id)
    
    if not success:
        raise PDFNotFoundError(f"PDF {file_id} not found")
    
    return {"success": True}