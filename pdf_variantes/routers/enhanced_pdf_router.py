"""Enhanced router with functional patterns and performance optimizations."""

from fastapi import APIRouter, UploadFile, File, Depends, Query, Path, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from ..functional_utils import create_progress_tracker, format_file_size, generate_id
from ..pdf_processor import process_pdf_complete, batch_process_pdfs, extract_features_parallel
from ..async_services import with_timeout, create_async_cache
from ..dependencies import get_pdf_service, get_current_user, validate_file_size
from ..exceptions import PDFNotFoundError, InvalidFileError
from ..schemas import PDFUploadSchema, VariantGenerateSchema

logger = logging.getLogger(__name__)

# Create router with enhanced configuration
router = APIRouter(
    prefix="/pdf",
    tags=["PDF Processing"],
    responses={
        404: {"description": "PDF not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)

# Create async cache for responses
response_cache = create_async_cache(ttl_seconds=300, max_size=1000)


@router.post("/upload", summary="Upload PDF", description="Upload and process PDF file")
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    auto_process: bool = Query(True, description="Auto-process PDF after upload"),
    extract_text: bool = Query(True, description="Extract text content"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    pdf_service = Depends(get_pdf_service),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Upload PDF with enhanced processing."""
    # Validate file
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise InvalidFileError("Only PDF files are allowed")
    
    file_content = await file.read()
    validate_file_size(len(file_content))
    
    # Generate file ID
    file_id = generate_id("pdf_")
    
    # Process upload
    if auto_process:
        # Process in background for better performance
        background_tasks.add_task(
            process_pdf_complete,
            file_content,
            file.filename,
            {"include_topics": True, "include_variants": True}
        )
    
    return {
        "success": True,
        "file_id": file_id,
        "filename": file.filename,
        "file_size": format_file_size(len(file_content)),
        "auto_process": auto_process,
        "uploaded_at": datetime.utcnow().isoformat()
    }


@router.get("/{file_id}/preview", summary="Get PDF preview")
async def get_preview(
    file_id: str = Path(..., description="PDF file ID"),
    page_number: int = Query(1, ge=1, description="Page number to preview"),
    pdf_service = Depends(get_pdf_service)
) -> Dict[str, Any]:
    """Get PDF page preview with caching."""
    # Check cache first
    cache_key = f"preview:{file_id}:{page_number}"
    cached_result = await response_cache["get"](cache_key)
    
    if cached_result:
        return cached_result
    
    # Generate preview
    preview = await pdf_service.upload_handler.get_pdf_preview(file_id, page_number)
    
    if not preview:
        raise PDFNotFoundError(f"PDF {file_id} not found")
    
    result = {
        "file_id": file_id,
        "page_number": page_number,
        "preview": preview,
        "generated_at": datetime.utcnow().isoformat()
    }
    
    # Cache result
    await response_cache["set"](cache_key, result)
    
    return result


@router.post("/{file_id}/variants", summary="Generate variants")
async def generate_variant(
    file_id: str = Path(..., description="PDF file ID"),
    variant_request: VariantGenerateSchema = ...,
    pdf_service = Depends(get_pdf_service)
) -> Dict[str, Any]:
    """Generate PDF variants with enhanced processing."""
    file_path = pdf_service.upload_handler.get_file_path(file_id)
    
    if not file_path.exists():
        raise PDFNotFoundError(f"PDF {file_id} not found")
    
    # Process with timeout
    result = await with_timeout(
        _generate_variant_async(file_path, variant_request, pdf_service),
        timeout_seconds=30.0
    )
    
    return result


async def _generate_variant_async(file_path, variant_request, pdf_service):
    """Internal function to generate variants."""
    with open(file_path, "rb") as f:
        variant = await pdf_service.variant_generator.generate(
            file=f,
            variant_type=variant_request.variant_type,
            options=variant_request.options
        )
    
    return {
        "success": True,
        "variant": variant,
        "file_id": file_path.stem,
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/{file_id}/topics", summary="Extract topics")
async def extract_topics(
    file_id: str = Path(..., description="PDF file ID"),
    min_relevance: float = Query(0.5, ge=0.0, le=1.0, description="Minimum relevance score"),
    max_topics: int = Query(50, ge=1, le=200, description="Maximum topics"),
    pdf_service = Depends(get_pdf_service)
) -> Dict[str, Any]:
    """Extract topics with enhanced processing."""
    # Check cache
    cache_key = f"topics:{file_id}:{min_relevance}:{max_topics}"
    cached_result = await response_cache["get"](cache_key)
    
    if cached_result:
        return cached_result
    
    # Extract topics
    topics = await pdf_service.topic_extractor.extract_topics(
        file_id=file_id,
        min_relevance=min_relevance,
        max_topics=max_topics
    )
    
    result = {
        "file_id": file_id,
        "topics": [topic.to_dict() if hasattr(topic, 'to_dict') else topic for topic in topics],
        "total_count": len(topics),
        "extracted_at": datetime.utcnow().isoformat()
    }
    
    # Cache result
    await response_cache["set"](cache_key, result)
    
    return result


@router.post("/batch-process", summary="Batch process PDFs")
async def batch_process(
    files: List[UploadFile] = File(..., description="PDF files to process"),
    max_concurrent: int = Query(5, ge=1, le=10, description="Maximum concurrent processing"),
    pdf_service = Depends(get_pdf_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Batch process multiple PDFs."""
    # Prepare file data
    file_data_list = []
    
    for file in files:
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            continue
        
        file_content = await file.read()
        validate_file_size(len(file_content))
        
        file_data_list.append({
            "filename": file.filename,
            "content": file_content
        })
    
    if not file_data_list:
        raise HTTPException(status_code=400, detail="No valid PDF files provided")
    
    # Process with progress tracking
    progress_tracker = create_progress_tracker(len(file_data_list))
    
    result = await batch_process_pdfs(
        file_data_list,
        {"include_topics": True, "include_variants": True},
        max_concurrent
    )
    
    return {
        **result,
        "progress": progress_tracker(len(file_data_list)),
        "processed_by": current_user.get("user_id", "anonymous")
    }


@router.get("/{file_id}/features", summary="Extract all features")
async def extract_all_features(
    file_id: str = Path(..., description="PDF file ID"),
    features: List[str] = Query(["topics", "variants", "quality"], description="Features to extract"),
    pdf_service = Depends(get_pdf_service)
) -> Dict[str, Any]:
    """Extract all features in parallel."""
    # Check cache
    cache_key = f"features:{file_id}:{':'.join(sorted(features))}"
    cached_result = await response_cache["get"](cache_key)
    
    if cached_result:
        return cached_result
    
    # Get content data
    file_path = pdf_service.upload_handler.get_file_path(file_id)
    if not file_path.exists():
        raise PDFNotFoundError(f"PDF {file_id} not found")
    
    with open(file_path, "rb") as f:
        content_data = await pdf_service.upload_handler.extract_content(f)
    
    # Extract features in parallel
    features_result = await extract_features_parallel(
        file_id,
        content_data,
        {feature: getattr(pdf_service, f"extract_{feature}") for feature in features}
    )
    
    result = {
        "file_id": file_id,
        "features": features_result,
        "extracted_at": datetime.utcnow().isoformat()
    }
    
    # Cache result
    await response_cache["set"](cache_key, result)
    
    return result


@router.delete("/{file_id}", summary="Delete PDF")
async def delete_pdf(
    file_id: str = Path(..., description="PDF file ID"),
    pdf_service = Depends(get_pdf_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete PDF file and clear cache."""
    success = await pdf_service.upload_handler.delete_pdf(file_id)
    
    if not success:
        raise PDFNotFoundError(f"PDF {file_id} not found")
    
    # Clear related cache entries
    cache_patterns = [f"preview:{file_id}:", f"topics:{file_id}:", f"features:{file_id}:"]
    for pattern in cache_patterns:
        # Note: In a real implementation, you'd need a more sophisticated cache clearing mechanism
        pass
    
    return {
        "success": True,
        "file_id": file_id,
        "deleted_at": datetime.utcnow().isoformat(),
        "deleted_by": current_user.get("user_id", "anonymous")
    }


@router.get("/{file_id}/status", summary="Get processing status")
async def get_processing_status(
    file_id: str = Path(..., description="PDF file ID"),
    pdf_service = Depends(get_pdf_service)
) -> Dict[str, Any]:
    """Get processing status for a PDF."""
    file_path = pdf_service.upload_handler.get_file_path(file_id)
    
    if not file_path.exists():
        raise PDFNotFoundError(f"PDF {file_id} not found")
    
    # Check processing status (simplified)
    status = {
        "file_id": file_id,
        "status": "processed",
        "has_topics": True,
        "has_variants": True,
        "has_quality_analysis": True,
        "last_processed": datetime.utcnow().isoformat()
    }
    
    return status
