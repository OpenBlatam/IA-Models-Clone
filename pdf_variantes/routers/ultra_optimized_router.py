"""
Ultra-Optimized Router
=====================

High-performance FastAPI router with streaming responses, background tasks,
early validation, and functional patterns.

Key Optimizations:
- Early validation with guard clauses
- Streaming responses for large files
- Background task processing
- Async operations throughout
- Error handling with early returns
- Caching integration
- Performance monitoring

Author: TruthGPT Development Team
Version: 1.0.0 - Ultra-Optimized
License: MIT
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, status
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional, Dict, Any, Union
import asyncio
import logging
import time
from io import BytesIO
import json

from .optimized_processor import (
    process_pdf_parallel, process_multiple_pdfs, ProcessingLimits,
    ProcessingResult, get_cache_stats, clear_cache, cleanup_cache
)
from .ultra_efficient_core import ProcessingConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/pdf-ultra", tags=["PDF Ultra Processing"])

# Processing limits
LIMITS = ProcessingLimits()

# Background task storage (in production, use Redis or database)
background_tasks: Dict[str, Dict[str, Any]] = {}

def validate_upload_file(file: UploadFile) -> Optional[str]:
    """Early validation with guard clauses."""
    if not file.filename:
        return "No filename provided"
    
    if not file.filename.lower().endswith('.pdf'):
        return "File must be a PDF"
    
    if file.size and file.size > LIMITS.max_file_size:
        return f"File too large: {file.size} bytes (max: {LIMITS.max_file_size})"
    
    return None

def validate_processing_params(
    operations: Optional[List[str]] = None,
    max_chars: Optional[int] = None,
    max_pages: Optional[int] = None
) -> Optional[str]:
    """Validate processing parameters with early returns."""
    if operations:
        valid_ops = {"text", "preview", "topics"}
        invalid_ops = set(operations) - valid_ops
        if invalid_ops:
            return f"Invalid operations: {invalid_ops}"
    
    if max_chars and max_chars > LIMITS.max_chars:
        return f"max_chars too large: {max_chars} (max: {LIMITS.max_chars})"
    
    if max_pages and max_pages > LIMITS.max_pages:
        return f"max_pages too large: {max_pages} (max: {LIMITS.max_pages})"
    
    return None

@router.post(
    "/upload",
    response_model=Dict[str, Any],
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process PDF with ultra-fast processing"
)
async def upload_pdf_ultra(
    file: UploadFile = File(..., description="PDF file to process"),
    operations: Optional[List[str]] = None,
    max_chars: Optional[int] = None,
    max_pages: Optional[int] = None,
    background: BackgroundTasks = BackgroundTasks()
) -> Dict[str, Any]:
    """
    Upload and process PDF with ultra-fast processing.
    
    Operations: text, preview, topics
    """
    # Early validation
    validation_error = validate_upload_file(file)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)
    
    param_error = validate_processing_params(operations, max_chars, max_pages)
    if param_error:
        raise HTTPException(status_code=400, detail=param_error)
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Create processing limits
        limits = ProcessingLimits(
            max_chars=max_chars or LIMITS.max_chars,
            max_pages=max_pages or LIMITS.max_pages
        )
        
        # Process PDF
        result = await process_pdf_parallel(file_content, operations, limits)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        # Add metadata
        response_data = {
            "success": True,
            "filename": file.filename,
            "file_size": len(file_content),
            "processing_time": result.processing_time,
            "operations": operations or ["text", "preview", "topics"],
            "data": result.data
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

@router.post(
    "/upload-stream",
    summary="Upload PDF with streaming response"
)
async def upload_pdf_stream(
    file: UploadFile = File(..., description="PDF file to process"),
    operations: Optional[List[str]] = None
) -> StreamingResponse:
    """
    Upload PDF with streaming response for large files.
    """
    # Early validation
    validation_error = validate_upload_file(file)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Process PDF
        result = await process_pdf_parallel(file_content, operations)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        # Create streaming response
        response_data = {
            "success": True,
            "filename": file.filename,
            "file_size": len(file_content),
            "processing_time": result.processing_time,
            "data": result.data
        }
        
        # Convert to JSON and stream
        json_data = json.dumps(response_data, default=str)
        
        def generate_chunks():
            chunk_size = 1024
            for i in range(0, len(json_data), chunk_size):
                yield json_data[i:i + chunk_size]
        
        return StreamingResponse(
            generate_chunks(),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={file.filename}_processed.json"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Streaming upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {e}")

@router.post(
    "/batch",
    response_model=Dict[str, Any],
    summary="Process multiple PDFs in batch"
)
async def process_batch_ultra(
    files: List[UploadFile] = File(..., description="Multiple PDF files"),
    operations: Optional[List[str]] = None,
    max_concurrent: int = 5
) -> Dict[str, Any]:
    """
    Process multiple PDFs in batch with controlled concurrency.
    """
    # Early validation
    if len(files) > 20:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many files (max: 20)")
    
    if max_concurrent > 10:
        raise HTTPException(status_code=400, detail="max_concurrent too high (max: 10)")
    
    # Validate all files
    for file in files:
        validation_error = validate_upload_file(file)
        if validation_error:
            raise HTTPException(status_code=400, detail=f"File {file.filename}: {validation_error}")
    
    try:
        # Prepare file data
        file_data = []
        for file in files:
            content = await file.read()
            file_data.append((file.filename, content))
        
        # Process batch
        start_time = time.time()
        results = await process_multiple_pdfs(file_data, operations, max_concurrent)
        total_time = time.time() - start_time
        
        # Process results
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        response_data = {
            "success": True,
            "total_files": len(files),
            "successful": successful,
            "failed": failed,
            "total_processing_time": total_time,
            "results": [
                {
                    "filename": file_data[i][0],
                    "success": result.success,
                    "processing_time": result.processing_time,
                    "data": result.data if result.success else None,
                    "error": result.error if not result.success else None
                }
                for i, result in enumerate(results)
            ]
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e}")

@router.get(
    "/preview/{file_hash}",
    summary="Get cached preview by file hash"
)
async def get_preview_cached(file_hash: str) -> StreamingResponse:
    """
    Get cached preview by file hash.
    """
    try:
        # This would typically query a cache or database
        # For now, return a placeholder
        placeholder_data = b"Preview not found"
        
        return StreamingResponse(
            BytesIO(placeholder_data),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"}
        )
        
    except Exception as e:
        logger.error(f"Preview retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preview retrieval failed: {e}")

@router.get(
    "/text/{file_hash}",
    response_model=Dict[str, Any],
    summary="Get cached text by file hash"
)
async def get_text_cached(file_hash: str) -> Dict[str, Any]:
    """
    Get cached text by file hash.
    """
    try:
        # This would typically query a cache or database
        # For now, return a placeholder
        return {
            "success": True,
            "file_hash": file_hash,
            "text": "Cached text not found",
            "cached": True
        }
        
    except Exception as e:
        logger.error(f"Text retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text retrieval failed: {e}")

@router.get(
    "/topics/{file_hash}",
    response_model=Dict[str, Any],
    summary="Get cached topics by file hash"
)
async def get_topics_cached(file_hash: str) -> Dict[str, Any]:
    """
    Get cached topics by file hash.
    """
    try:
        # This would typically query a cache or database
        # For now, return a placeholder
        return {
            "success": True,
            "file_hash": file_hash,
            "topics": ["cached", "topics", "not", "found"],
            "cached": True
        }
        
    except Exception as e:
        logger.error(f"Topics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Topics retrieval failed: {e}")

@router.post(
    "/background-process",
    response_model=Dict[str, Any],
    summary="Start background processing task"
)
async def start_background_process(
    file: UploadFile = File(..., description="PDF file to process"),
    operations: Optional[List[str]] = None,
    background: BackgroundTasks = BackgroundTasks()
) -> Dict[str, Any]:
    """
    Start background processing task.
    """
    # Early validation
    validation_error = validate_upload_file(file)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)
    
    try:
        # Generate task ID
        task_id = f"task_{int(time.time())}_{file.filename}"
        
        # Store task info
        background_tasks[task_id] = {
            "status": "pending",
            "filename": file.filename,
            "operations": operations or ["text", "preview", "topics"],
            "created_at": time.time(),
            "result": None,
            "error": None
        }
        
        # Add background task
        background.add_task(process_background_task, task_id, file)
        
        return {
            "success": True,
            "task_id": task_id,
            "status": "pending",
            "message": "Background processing started"
        }
        
    except Exception as e:
        logger.error(f"Background task creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Background task creation failed: {e}")

async def process_background_task(task_id: str, file: UploadFile):
    """Background task processor."""
    try:
        # Update status
        background_tasks[task_id]["status"] = "processing"
        
        # Read file content
        file_content = await file.read()
        
        # Process PDF
        result = await process_pdf_parallel(file_content, background_tasks[task_id]["operations"])
        
        # Update task result
        background_tasks[task_id]["status"] = "completed" if result.success else "failed"
        background_tasks[task_id]["result"] = result.data if result.success else None
        background_tasks[task_id]["error"] = result.error if not result.success else None
        background_tasks[task_id]["processing_time"] = result.processing_time
        
    except Exception as e:
        logger.error(f"Background task {task_id} failed: {e}")
        background_tasks[task_id]["status"] = "failed"
        background_tasks[task_id]["error"] = str(e)

@router.get(
    "/task/{task_id}",
    response_model=Dict[str, Any],
    summary="Get background task status"
)
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get background task status.
    """
    if task_id not in background_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = background_tasks[task_id]
    
    return {
        "success": True,
        "task_id": task_id,
        "status": task_info["status"],
        "filename": task_info["filename"],
        "operations": task_info["operations"],
        "created_at": task_info["created_at"],
        "result": task_info["result"],
        "error": task_info["error"],
        "processing_time": task_info.get("processing_time")
    }

@router.get(
    "/cache/stats",
    response_model=Dict[str, Any],
    summary="Get cache statistics"
)
async def get_cache_statistics() -> Dict[str, Any]:
    """
    Get cache statistics for monitoring.
    """
    try:
        stats = get_cache_stats()
        return {
            "success": True,
            "cache_stats": stats,
            "background_tasks": len(background_tasks)
        }
        
    except Exception as e:
        logger.error(f"Cache stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache stats retrieval failed: {e}")

@router.post(
    "/cache/clear",
    response_model=Dict[str, Any],
    summary="Clear cache"
)
async def clear_cache_endpoint() -> Dict[str, Any]:
    """
    Clear all cache entries.
    """
    try:
        cleared_count = cleanup_cache()
        clear_cache()
        
        return {
            "success": True,
            "message": f"Cache cleared successfully",
            "cleared_entries": cleared_count
        }
        
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {e}")

@router.get(
    "/health",
    response_model=Dict[str, Any],
    summary="Health check for ultra router"
)
async def health_check_ultra() -> Dict[str, Any]:
    """
    Health check for ultra router.
    """
    try:
        cache_stats = get_cache_stats()
        
        return {
            "success": True,
            "status": "healthy",
            "router": "pdf-ultra",
            "cache_stats": cache_stats,
            "background_tasks": len(background_tasks),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")
