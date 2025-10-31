from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, status, Path, FastAPI, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from core.config import get_config
from core.exceptions import OSContentException, ValidationError, ProcessingError, create_error_response
from core.types import ProcessingStatus
from services.video_service import video_service
from services.nlp_service import nlp_service
from services.file_service import file_service
from services.validation_service import validation_service
from performance_monitor import monitor, monitor_request
from cache_manager import cache
from async_processor import processor
from load_balancer import load_balancer
from cdn_manager import cdn_manager
import structlog
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Refactored API for OS Content UGC Video Generator
Uses service layer and core components for better separation of concerns
"""



logger = structlog.get_logger("os_content.api")

app = FastAPI()
config = get_config()

# Rate limiting
rate_limit_store = {}

def cleanup_rate_limit_store():
    """Clean up old rate limit entries"""
    current_time = datetime.utcnow().timestamp()
    for ip in list(rate_limit_store.keys()):
        rate_limit_store[ip] = [
            timestamp for timestamp in rate_limit_store[ip] 
            if current_time - timestamp < config.security.rate_limit_window
        ]
        if not rate_limit_store[ip]:
            del rate_limit_store[ip]

def check_rate_limit(client_ip: str) -> bool:
    """Check if client is within rate limits"""
    cleanup_rate_limit_store()
    current_time = datetime.utcnow().timestamp()
    
    if len(rate_limit_store.get(client_ip, [])) >= config.security.rate_limit:
        return False
    
    if client_ip not in rate_limit_store:
        rate_limit_store[client_ip] = []
    rate_limit_store[client_ip].append(current_time)
    return True

@app.middleware("http")
async def optimized_middleware(request: Request, call_next):
    """Optimized middleware with error handling"""
    start_time = datetime.utcnow().timestamp()
    client_ip = request.client.host if request.client else "unknown"
    
    # Rate limiting
    if not check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content=create_error_response(
                ValidationError("Rate limit exceeded", violation_type="rate_limit", client_ip=client_ip)
            )
        )
    
    try:
        response = await call_next(request)
        process_time = datetime.utcnow().timestamp() - start_time
        
        # Record performance metrics
        monitor.record_request(process_time, success=True)
        
        # Add performance headers
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        response.headers["X-Rate-Limit-Remaining"] = str(
            config.security.rate_limit - len(rate_limit_store.get(client_ip, []))
        )
        
        logger.info(
            f"{request.method} {request.url.path} completed in {process_time:.3f}s",
            extra={
                "module": __name__,
                "path": request.url.path,
                "method": request.method,
                "process_time": process_time,
                "client_ip": client_ip
            }
        )
        return response
        
    except OSContentException as exc:
        process_time = datetime.utcnow().timestamp() - start_time
        monitor.record_request(process_time, success=False)
        
        logger.warning(
            f"OSContentException: {exc.message}",
            extra={
                "module": __name__,
                "path": request.url.path,
                "method": request.method,
                "error_code": exc.error_code,
                "client_ip": client_ip
            }
        )
        
        return JSONResponse(
            status_code=400 if isinstance(exc, ValidationError) else 500,
            content=create_error_response(exc)
        )
        
    except Exception as exc:
        process_time = datetime.utcnow().timestamp() - start_time
        monitor.record_request(process_time, success=False)
        
        logger.error(
            f"Unhandled Exception: {exc}",
            extra={
                "module": __name__,
                "path": request.url.path,
                "method": request.method,
                "client_ip": client_ip
            }
        )
        
        return JSONResponse(
            status_code=500,
            content=create_error_response(
                OSContentException(f"Internal server error: {str(exc)}", "INTERNAL_ERROR")
            )
        )

router = APIRouter(prefix="/os-content", tags=["OS Content UGC Video"])

@router.on_event('startup')
async def on_startup() -> None:
    """Initialize resources on startup"""
    logger.info("OS Content API started successfully")

@router.on_event('shutdown')
async def on_shutdown() -> None:
    """Cleanup resources on shutdown"""
    logger.info("OS Content API shutdown complete")

@router.post("/ugc-video", status_code=status.HTTP_202_ACCEPTED)
@monitor_request
async def create_ugc_video(
    background_tasks: BackgroundTasks,
    user_id: str = Form(..., description="ID del usuario solicitante"),
    title: str = Form(..., description="Título del video"),
    text_prompt: str = Form(..., description="Texto o guion para el video"),
    description: Optional[str] = Form(None, description="Descripción opcional"),
    ugc_type: str = Form("ugc_video_ad", description="Tipo de UGC"),
    images: Optional[List[UploadFile]] = File(None),
    videos: Optional[List[UploadFile]] = File(None),
    language: str = Form("es", description="Idioma principal del texto"),
    duration_per_image: float = Form(3.0, description="Duración por imagen en segundos"),
):
    """Create a new UGC video processing request"""
    
    try:
        # Process uploaded files
        image_files = []
        video_files = []
        
        if images:
            for img in images:
                if img.content_type and not img.content_type.startswith('image/'):
                    raise ValidationError(f"File {img.filename} is not a valid image", field="content_type")
                content = await img.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                file_path = await file_service.save_uploaded_file(img.filename, content)
                image_files.append(file_path)
        
        if videos:
            for vid in videos:
                if vid.content_type and not vid.content_type.startswith('video/'):
                    raise ValidationError(f"File {vid.filename} is not a valid video", field="content_type")
                content = await vid.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                file_path = await file_service.save_uploaded_file(vid.filename, content)
                video_files.append(file_path)
        
        # Create video using service
        response = await video_service.create_video(
            user_id=user_id,
            title=title,
            text_prompt=text_prompt,
            image_files=image_files,
            video_files=video_files,
            language=language,
            duration_per_image=duration_per_image,
            description=description,
            ugc_type=ugc_type
        )
        
        return response
        
    except OSContentException:
        raise
    except Exception as e:
        logger.error(f"Error creating video: {e}")
        raise OSContentException(f"Failed to create video: {str(e)}", "VIDEO_CREATION_ERROR")

@router.get("/ugc-video/{request_id}/status")
@monitor_request
async def get_ugc_video_status(request_id: str = Path(..., description="ID de la solicitud de video UGC")):
    """Get the status of a UGC video request"""
    
    try:
        response = await video_service.get_video_status(request_id)
        return response
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error getting video status: {e}")
        raise OSContentException(f"Failed to get video status: {str(e)}", "VIDEO_STATUS_ERROR")

@router.post("/nlp")
@monitor_request
async def nlp_endpoint(text: str = Form(..., description="Text to analyze"), lang: str = Form("es", description="Language")):
    """Analyze text using NLP"""
    
    try:
        result = await nlp_service.analyze_text(text, lang)
        return result
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error in NLP analysis: {e}")
        raise OSContentException(f"Failed to analyze text: {str(e)}", "NLP_ANALYSIS_ERROR")

@router.post("/nlp/batch")
@monitor_request
async def batch_nlp_endpoint(texts: List[str], lang: str = "es"):
    """Batch NLP analysis for multiple texts"""
    
    try:
        results = await nlp_service.batch_analyze_texts(texts, lang)
        return results
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error in batch NLP analysis: {e}")
        raise OSContentException(f"Failed to analyze texts: {str(e)}", "BATCH_NLP_ERROR")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@router.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    return {
        "uptime": monitor.get_uptime(),
        "total_requests": monitor.request_count,
        "success_rate": monitor.get_success_rate(),
        "average_processing_time": monitor.get_average_processing_time(),
        "error_count": monitor.error_count,
        "current_metrics": monitor.get_system_metrics().__dict__ if monitor.metrics_history else None
    }

@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    return cache.get_stats()

@router.get("/processor/stats")
async def get_processor_stats():
    """Get async processor statistics"""
    return processor.get_stats()

@router.get("/load-balancer/stats")
async def get_load_balancer_stats():
    """Get load balancer statistics"""
    return load_balancer.get_stats()

@router.get("/cdn/stats")
async def get_cdn_stats():
    """Get CDN manager statistics"""
    return cdn_manager.get_stats()

@router.get("/nlp/stats")
async def get_nlp_stats():
    """Get NLP service statistics"""
    return await nlp_service.get_nlp_stats()

@router.get("/storage/stats")
async def get_storage_stats():
    """Get storage statistics"""
    return await file_service.get_storage_stats()

@router.get("/video/stats")
async def get_video_stats():
    """Get video processing statistics"""
    return await video_service.get_processing_stats()

@router.post("/cache/clear")
async def clear_cache():
    """Clear all caches"""
    await cache.clear()
    return {"message": "Cache cleared successfully"}

@router.post("/cdn/cleanup")
async def cleanup_cdn_cache():
    """Cleanup CDN cache"""
    await cdn_manager.cleanup_cache()
    return {"message": "CDN cache cleaned up successfully"}

@router.post("/storage/cleanup")
async def cleanup_storage():
    """Cleanup temporary files"""
    deleted_count = await file_service.cleanup_temp_files()
    return {"message": f"Cleaned up {deleted_count} temporary files"}

@router.delete("/ugc-video/{request_id}")
async def cancel_video(request_id: str = Path(..., description="ID de la solicitud de video UGC")):
    """Cancel video processing"""
    
    try:
        success = await video_service.cancel_video(request_id)
        return {"message": "Video processing cancelled successfully" if success else "Failed to cancel video"}
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error cancelling video: {e}")
        raise OSContentException(f"Failed to cancel video: {str(e)}", "VIDEO_CANCELLATION_ERROR")

# Include router in app
app.include_router(router) 