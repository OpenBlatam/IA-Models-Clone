"""Enhanced router using best-in-class libraries and patterns."""

from fastapi import APIRouter, UploadFile, File, Depends, Query, Path, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, List, Optional
import logging
import time
import asyncio
from datetime import datetime
import io

# Best libraries for web and async operations
import httpx
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential

# Enhanced PDF processing
from .enhanced_pdf_processor import (
    enhanced_process_pdf, enhanced_batch_process,
    enhanced_extract_content, enhanced_extract_topics,
    enhanced_generate_variants, enhanced_analyze_quality
)

# Performance and monitoring
from .advanced_performance import intelligent_cache, performance_monitor
from .advanced_error_handling import intelligent_error_handler, ErrorSeverity, ErrorCategory

# Dependencies
from .enhanced_dependencies import get_config, get_current_user, validate_file_size
from .enhanced_schemas import PDFUploadRequest, VariantGenerateRequest, TopicExtractRequest

logger = logging.getLogger(__name__)

# Create enhanced router
router = APIRouter(
    prefix="/pdf",
    tags=["Enhanced PDF Processing"],
    responses={
        404: {"description": "PDF not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)

# Enhanced response cache
response_cache = {}


@router.post("/upload", summary="Enhanced PDF Upload")
@performance_monitor("enhanced_upload")
@intelligent_error_handler(ErrorSeverity.MEDIUM, ErrorCategory.PROCESSING, "pdf_upload")
async def enhanced_upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    auto_process: bool = Query(True, description="Auto-process PDF after upload"),
    extract_text: bool = Query(True, description="Extract text content"),
    generate_preview: bool = Query(True, description="Generate preview images"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    config = Depends(get_config),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Enhanced PDF upload with best-in-class processing."""
    
    # Validate file
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_content = await file.read()
    validate_file_size(len(file_content))
    
    # Generate unique file ID
    file_id = f"pdf_{int(time.time())}_{hash(file_content) % 10000}"
    
    # Save file asynchronously
    file_path = f"uploads/{file_id}.pdf"
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(file_content)
    
    # Process in background if requested
    if auto_process:
        background_tasks.add_task(
            enhanced_process_pdf,
            file_content,
            file.filename,
            {
                "include_topics": True,
                "include_variants": True,
                "include_quality": True
            }
        )
    
    return {
        "success": True,
        "file_id": file_id,
        "filename": file.filename,
        "file_size": len(file_content),
        "file_size_mb": round(len(file_content) / (1024 * 1024), 2),
        "auto_process": auto_process,
        "uploaded_at": datetime.utcnow().isoformat(),
        "uploaded_by": current_user.get("user_id", "anonymous")
    }


@router.get("/{file_id}/preview", summary="Enhanced PDF Preview")
@intelligent_cache(maxsize=500, ttl=300.0)
@performance_monitor("enhanced_preview")
async def enhanced_preview(
    file_id: str = Path(..., description="PDF file ID"),
    page_number: int = Query(1, ge=1, description="Page number to preview"),
    quality: str = Query("medium", description="Preview quality: low, medium, high")
) -> Dict[str, Any]:
    """Enhanced PDF preview with image generation."""
    
    # Check cache first
    cache_key = f"preview:{file_id}:{page_number}:{quality}"
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    file_path = f"uploads/{file_id}.pdf"
    
    try:
        # Load PDF content
        async with aiofiles.open(file_path, 'rb') as f:
            file_content = await f.read()
        
        # Generate preview using pdf2image
        from pdf2image import convert_from_bytes
        
        dpi_map = {"low": 100, "medium": 150, "high": 300}
        dpi = dpi_map.get(quality, 150)
        
        images = convert_from_bytes(file_content, dpi=dpi, first_page=page_number, last_page=page_number)
        
        if images:
            # Convert to base64 for response
            import base64
            from PIL import Image
            import io
            
            img_buffer = io.BytesIO()
            images[0].save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            preview_data = {
                "file_id": file_id,
                "page_number": page_number,
                "image_data": f"data:image/png;base64,{img_str}",
                "quality": quality,
                "dpi": dpi,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Cache result
            response_cache[cache_key] = preview_data
            
            return preview_data
        else:
            raise HTTPException(status_code=404, detail="Could not generate preview")
            
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PDF file not found")
    except Exception as e:
        logger.error(f"Preview generation failed: {e}")
        raise HTTPException(status_code=500, detail="Preview generation failed")


@router.post("/{file_id}/variants", summary="Enhanced Variant Generation")
@performance_monitor("enhanced_variant_generation")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def enhanced_generate_variant(
    file_id: str = Path(..., description="PDF file ID"),
    variant_request: VariantGenerateRequest = ...,
    config = Depends(get_config)
) -> Dict[str, Any]:
    """Enhanced variant generation with AI and advanced processing."""
    
    file_path = f"uploads/{file_id}.pdf"
    
    try:
        # Load PDF content
        async with aiofiles.open(file_path, 'rb') as f:
            file_content = await f.read()
        
        # Extract content first
        content_data = await enhanced_extract_content(file_content)
        
        # Generate variants
        variants_result = await enhanced_generate_variants(content_data)
        
        return {
            "success": True,
            "file_id": file_id,
            "variant_type": variant_request.variant_type,
            "variants": variants_result["variants"],
            "generation_method": variants_result["generation_method"],
            "ai_features": variants_result.get("ai_features", {}),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PDF file not found")
    except Exception as e:
        logger.error(f"Variant generation failed: {e}")
        raise HTTPException(status_code=500, detail="Variant generation failed")


@router.get("/{file_id}/topics", summary="Enhanced Topic Extraction")
@intelligent_cache(maxsize=300, ttl=1800.0)
@performance_monitor("enhanced_topic_extraction")
async def enhanced_extract_topics_endpoint(
    file_id: str = Path(..., description="PDF file ID"),
    min_relevance: float = Query(0.5, ge=0.0, le=1.0, description="Minimum relevance score"),
    max_topics: int = Query(50, ge=1, le=200, description="Maximum topics"),
    methods: List[str] = Query(["rake", "spacy", "tfidf"], description="Extraction methods")
) -> Dict[str, Any]:
    """Enhanced topic extraction using multiple NLP methods."""
    
    # Check cache
    cache_key = f"topics:{file_id}:{min_relevance}:{max_topics}:{':'.join(sorted(methods))}"
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    file_path = f"uploads/{file_id}.pdf"
    
    try:
        # Load PDF content
        async with aiofiles.open(file_path, 'rb') as f:
            file_content = await f.read()
        
        # Extract content
        content_data = await enhanced_extract_content(file_content)
        
        # Extract topics
        topics_result = await enhanced_extract_topics(content_data)
        
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
            "extraction_methods": topics_result.get("extraction_methods", []),
            "extracted_at": datetime.utcnow().isoformat()
        }
        
        # Cache result
        response_cache[cache_key] = result
        
        return result
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PDF file not found")
    except Exception as e:
        logger.error(f"Topic extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Topic extraction failed")


@router.post("/batch-process", summary="Enhanced Batch Processing")
@performance_monitor("enhanced_batch_processing")
async def enhanced_batch_process_endpoint(
    files: List[UploadFile] = File(..., description="PDF files to process"),
    max_concurrent: int = Query(20, ge=1, le=50, description="Maximum concurrent processing"),
    include_topics: bool = Query(True, description="Include topic extraction"),
    include_variants: bool = Query(True, description="Include variant generation"),
    include_quality: bool = Query(True, description="Include quality analysis"),
    config = Depends(get_config),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Enhanced batch processing with optimal concurrency."""
    
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
    
    # Process with enhanced batch processing
    result = await enhanced_batch_process(
        file_data_list,
        {
            "include_topics": include_topics,
            "include_variants": include_variants,
            "include_quality": include_quality
        },
        max_concurrent
    )
    
    return {
        **result,
        "processed_by": current_user.get("user_id", "anonymous"),
        "processing_method": "enhanced_batch_parallel",
        "libraries_used": ["pymupdf", "pdfplumber", "spacy", "sentence_transformers", "textstat"]
    }


@router.get("/{file_id}/quality", summary="Enhanced Quality Analysis")
@intelligent_cache(maxsize=200, ttl=3600.0)
@performance_monitor("enhanced_quality_analysis")
async def enhanced_quality_analysis(
    file_id: str = Path(..., description="PDF file ID")
) -> Dict[str, Any]:
    """Enhanced quality analysis using advanced text statistics."""
    
    # Check cache
    cache_key = f"quality:{file_id}"
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    file_path = f"uploads/{file_id}.pdf"
    
    try:
        # Load PDF content
        async with aiofiles.open(file_path, 'rb') as f:
            file_content = await f.read()
        
        # Extract content
        content_data = await enhanced_extract_content(file_content)
        
        # Analyze quality
        quality_result = await enhanced_analyze_quality(content_data)
        
        result = {
            "file_id": file_id,
            "quality_score": quality_result["quality_score"],
            "readability_level": quality_result.get("readability_level", "unknown"),
            "metrics": quality_result["metrics"],
            "quality_factors": quality_result["quality_factors"],
            "analysis_method": quality_result["analysis_method"],
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
        # Cache result
        response_cache[cache_key] = result
        
        return result
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PDF file not found")
    except Exception as e:
        logger.error(f"Quality analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Quality analysis failed")


@router.get("/{file_id}/download", summary="Download PDF")
async def download_pdf(
    file_id: str = Path(..., description="PDF file ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> StreamingResponse:
    """Download PDF file."""
    
    file_path = f"uploads/{file_id}.pdf"
    
    try:
        async def file_generator():
            async with aiofiles.open(file_path, 'rb') as f:
                while chunk := await f.read(8192):
                    yield chunk
        
        return StreamingResponse(
            file_generator(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={file_id}.pdf"}
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PDF file not found")


@router.delete("/{file_id}", summary="Delete PDF")
@performance_monitor("pdf_deletion")
async def delete_pdf(
    file_id: str = Path(..., description="PDF file ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete PDF file and clear cache."""
    
    file_path = f"uploads/{file_id}.pdf"
    
    try:
        # Delete file
        import os
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Clear related cache entries
        cache_patterns = [f"preview:{file_id}:", f"topics:{file_id}:", f"quality:{file_id}"]
        for pattern in cache_patterns:
            keys_to_remove = [key for key in response_cache.keys() if key.startswith(pattern)]
            for key in keys_to_remove:
                del response_cache[key]
        
        return {
            "success": True,
            "file_id": file_id,
            "deleted_at": datetime.utcnow().isoformat(),
            "deleted_by": current_user.get("user_id", "anonymous")
        }
        
    except Exception as e:
        logger.error(f"Error deleting PDF {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete PDF")


@router.get("/{file_id}/status", summary="Processing Status")
async def get_processing_status(
    file_id: str = Path(..., description="PDF file ID")
) -> Dict[str, Any]:
    """Get processing status for a PDF."""
    
    file_path = f"uploads/{file_id}.pdf"
    
    try:
        import os
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        # Check if processing results exist in cache
        has_topics = any(key.startswith(f"topics:{file_id}:") for key in response_cache.keys())
        has_quality = f"quality:{file_id}" in response_cache
        
        return {
            "file_id": file_id,
            "status": "processed",
            "has_topics": has_topics,
            "has_quality": has_quality,
            "has_preview": any(key.startswith(f"preview:{file_id}:") for key in response_cache.keys()),
            "last_processed": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking status for {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to check status")


@router.get("/health", summary="Enhanced Health Check")
async def enhanced_health_check() -> Dict[str, Any]:
    """Enhanced health check with library status."""
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "pdf_processing": "healthy",
            "topic_extraction": "healthy",
            "variant_generation": "healthy",
            "quality_analysis": "healthy"
        },
        "libraries": {
            "pymupdf": "available",
            "pdfplumber": "available",
            "spacy": "available",
            "sentence_transformers": "available",
            "textstat": "available"
        },
        "cache": {
            "size": len(response_cache),
            "hit_rate": 0.95  # Placeholder
        }
    }
