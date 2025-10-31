"""Optimized router with functional patterns and best practices."""

from fastapi import APIRouter, UploadFile, File, Depends, Query, Path, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, List, Optional
import logging
import time
import asyncio
from datetime import datetime
import os

# Core libraries
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential

# Optimized processor
from .optimized_processor import (
    process_pdf_optimized, batch_process_optimized,
    validate_pdf_content, extract_text_with_fallback
)

# Dependencies and schemas
from .enhanced_dependencies import get_config, get_current_user, validate_file_size
from .enhanced_schemas import PDFUploadRequest, VariantGenerateRequest, TopicExtractRequest

logger = logging.getLogger(__name__)

# Create optimized router
router = APIRouter(
    prefix="/pdf",
    tags=["Optimized PDF Processing"],
    responses={
        404: {"description": "PDF not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)

# In-memory cache for responses
response_cache: Dict[str, Any] = {}


def validate_file_early(file: UploadFile) -> None:
    """Early validation with guard clauses."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    if len(file.filename) > 255:
        raise HTTPException(status_code=400, detail="Filename too long")


def generate_file_id() -> str:
    """Generate unique file ID."""
    return f"pdf_{int(time.time())}_{hash(time.time()) % 10000}"


def format_file_size(size_bytes: int) -> str:
    """Format file size with early returns."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"


async def save_file_async(file_content: bytes, file_path: str) -> bool:
    """Save file asynchronously with error handling."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        return True
    except Exception as e:
        logger.error(f"Failed to save file {file_path}: {e}")
        return False


async def load_file_async(file_path: str) -> Optional[bytes]:
    """Load file asynchronously with error handling."""
    try:
        if not os.path.exists(file_path):
            return None
        
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()
    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {e}")
        return None


def clear_cache_pattern(pattern: str) -> None:
    """Clear cache entries matching pattern."""
    keys_to_remove = [key for key in response_cache.keys() if key.startswith(pattern)]
    for key in keys_to_remove:
        del response_cache[key]


@router.post("/upload", summary="Optimized PDF Upload")
async def upload_pdf_optimized(
    file: UploadFile = File(..., description="PDF file to upload"),
    auto_process: bool = Query(True, description="Auto-process PDF after upload"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    config = Depends(get_config),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Optimized PDF upload with functional patterns."""
    
    # Early validation
    validate_file_early(file)
    
    # Read file content
    file_content = await file.read()
    
    # Validate file size
    validate_file_size(len(file_content))
    
    # Validate PDF content
    validation = validate_pdf_content(file_content)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=validation["error"])
    
    # Generate file ID and path
    file_id = generate_file_id()
    file_path = f"uploads/{file_id}.pdf"
    
    # Save file
    save_success = await save_file_async(file_content, file_path)
    if not save_success:
        raise HTTPException(status_code=500, detail="Failed to save file")
    
    # Process in background if requested
    if auto_process:
        background_tasks.add_task(process_pdf_optimized, file_content, file.filename)
    
    return {
        "success": True,
        "file_id": file_id,
        "filename": file.filename,
        "file_size": format_file_size(len(file_content)),
        "auto_process": auto_process,
        "uploaded_at": datetime.utcnow().isoformat(),
        "uploaded_by": current_user.get("user_id", "anonymous") if current_user else "anonymous"
    }


@router.get("/{file_id}/preview", summary="PDF Preview")
async def get_preview(
    file_id: str = Path(..., description="PDF file ID"),
    page_number: int = Query(1, ge=1, description="Page number to preview")
) -> Dict[str, Any]:
    """Get PDF preview with caching."""
    
    # Check cache first
    cache_key = f"preview:{file_id}:{page_number}"
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    file_path = f"uploads/{file_id}.pdf"
    file_content = await load_file_async(file_path)
    
    if not file_content:
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    # Generate preview (simplified for this example)
    preview_data = {
        "file_id": file_id,
        "page_number": page_number,
        "preview_text": f"Preview for page {page_number}",
        "generated_at": datetime.utcnow().isoformat()
    }
    
    # Cache result
    response_cache[cache_key] = preview_data
    
    return preview_data


@router.post("/{file_id}/process", summary="Process PDF")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def process_pdf(
    file_id: str = Path(..., description="PDF file ID"),
    include_topics: bool = Query(True, description="Include topic extraction"),
    include_variants: bool = Query(True, description="Include variant generation"),
    include_quality: bool = Query(True, description="Include quality analysis")
) -> Dict[str, Any]:
    """Process PDF with specified features."""
    
    file_path = f"uploads/{file_id}.pdf"
    file_content = await load_file_async(file_path)
    
    if not file_content:
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    # Process PDF
    result = await process_pdf_optimized(file_content, f"{file_id}.pdf")
    
    # Filter results based on options
    filtered_result = {
        "file_id": result["file_id"],
        "processed_at": result["processed_at"]
    }
    
    if include_topics:
        filtered_result["topics"] = result["topics"]
    if include_variants:
        filtered_result["variants"] = result["variants"]
    if include_quality:
        filtered_result["quality"] = result["quality"]
    
    return filtered_result


@router.get("/{file_id}/topics", summary="Extract Topics")
async def extract_topics(
    file_id: str = Path(..., description="PDF file ID"),
    min_relevance: float = Query(0.5, ge=0.0, le=1.0, description="Minimum relevance score"),
    max_topics: int = Query(50, ge=1, le=200, description="Maximum topics")
) -> Dict[str, Any]:
    """Extract topics from PDF."""
    
    # Check cache
    cache_key = f"topics:{file_id}:{min_relevance}:{max_topics}"
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    file_path = f"uploads/{file_id}.pdf"
    file_content = await load_file_async(file_path)
    
    if not file_content:
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    # Extract text
    text = extract_text_with_fallback(file_content)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    # Extract topics
    from .optimized_processor import extract_topics_parallel
    topics_result = await extract_topics_parallel(text)
    
    # Filter by relevance
    filtered_topics = [
        topic for topic in topics_result["topics"]
        if topic.get("confidence", 0) >= min_relevance
    ][:max_topics]
    
    result = {
        "file_id": file_id,
        "topics": filtered_topics,
        "total_count": len(filtered_topics),
        "main_topic": topics_result.get("main_topic"),
        "confidence": topics_result.get("confidence", 0),
        "extracted_at": datetime.utcnow().isoformat()
    }
    
    # Cache result
    response_cache[cache_key] = result
    
    return result


@router.post("/batch-process", summary="Batch Process PDFs")
async def batch_process_pdfs(
    files: List[UploadFile] = File(..., description="PDF files to process"),
    max_concurrent: int = Query(20, ge=1, le=50, description="Maximum concurrent processing"),
    config = Depends(get_config),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Batch process multiple PDFs."""
    
    # Prepare file data with early validation
    file_data_list = []
    
    for file in files:
        validate_file_early(file)
        
        file_content = await file.read()
        validate_file_size(len(file_content))
        
        validation = validate_pdf_content(file_content)
        if not validation["valid"]:
            continue  # Skip invalid files
        
        file_data_list.append({
            "filename": file.filename,
            "content": file_content
        })
    
    if not file_data_list:
        raise HTTPException(status_code=400, detail="No valid PDF files provided")
    
    # Process files
    result = await batch_process_optimized(file_data_list, max_concurrent)
    
    return {
        **result,
        "processed_by": current_user.get("user_id", "anonymous"),
        "processing_method": "optimized_batch_parallel"
    }


@router.get("/{file_id}/quality", summary="Quality Analysis")
async def analyze_quality(
    file_id: str = Path(..., description="PDF file ID")
) -> Dict[str, Any]:
    """Analyze PDF quality."""
    
    # Check cache
    cache_key = f"quality:{file_id}"
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    file_path = f"uploads/{file_id}.pdf"
    file_content = await load_file_async(file_path)
    
    if not file_content:
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    # Extract text and analyze
    text = extract_text_with_fallback(file_content)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    from .optimized_processor import analyze_quality_comprehensive
    quality_result = await analyze_quality_comprehensive(text)
    
    result = {
        "file_id": file_id,
        "quality_score": quality_result["quality_score"],
        "metrics": quality_result["metrics"],
        "quality_factors": quality_result["quality_factors"],
        "analyzed_at": datetime.utcnow().isoformat()
    }
    
    # Cache result
    response_cache[cache_key] = result
    
    return result


@router.get("/{file_id}/download", summary="Download PDF")
async def download_pdf(
    file_id: str = Path(..., description="PDF file ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> StreamingResponse:
    """Download PDF file."""
    
    file_path = f"uploads/{file_id}.pdf"
    file_content = await load_file_async(file_path)
    
    if not file_content:
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    async def file_generator():
        chunk_size = 8192
        for i in range(0, len(file_content), chunk_size):
            yield file_content[i:i + chunk_size]
    
    return StreamingResponse(
        file_generator(),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={file_id}.pdf"}
    )


@router.delete("/{file_id}", summary="Delete PDF")
async def delete_pdf(
    file_id: str = Path(..., description="PDF file ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete PDF file and clear cache."""
    
    file_path = f"uploads/{file_id}.pdf"
    
    # Delete file
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    # Clear related cache entries
    clear_cache_pattern(f"preview:{file_id}:")
    clear_cache_pattern(f"topics:{file_id}:")
    clear_cache_pattern(f"quality:{file_id}")
    
    return {
        "success": True,
        "file_id": file_id,
        "deleted_at": datetime.utcnow().isoformat(),
        "deleted_by": current_user.get("user_id", "anonymous")
    }


@router.get("/{file_id}/status", summary="Processing Status")
async def get_status(
    file_id: str = Path(..., description="PDF file ID")
) -> Dict[str, Any]:
    """Get processing status for PDF."""
    
    file_path = f"uploads/{file_id}.pdf"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    # Check cache for processing results
    has_topics = any(key.startswith(f"topics:{file_id}:") for key in response_cache.keys())
    has_quality = f"quality:{file_id}" in response_cache
    has_preview = any(key.startswith(f"preview:{file_id}:") for key in response_cache.keys())
    
    return {
        "file_id": file_id,
        "status": "processed",
        "has_topics": has_topics,
        "has_quality": has_quality,
        "has_preview": has_preview,
        "last_checked": datetime.utcnow().isoformat()
    }


@router.get("/health", summary="Health Check")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "pdf_processing": "healthy",
            "topic_extraction": "healthy",
            "variant_generation": "healthy",
            "quality_analysis": "healthy"
        },
        "cache": {
            "size": len(response_cache),
            "hit_rate": 0.95  # Placeholder
        }
    }
