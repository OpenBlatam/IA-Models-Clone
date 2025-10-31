from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import logging
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, status, Path, FastAPI, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Dict, Any
from .models import OSContentUGCVideoRequest, OSContentUGCVideoResponse, NLPRequest, NLPResponse
from uuid import uuid4
from datetime import datetime
import os
from .video_pipeline import create_ugc_video_ad_with_langchain
from .nlp_utils import analyze_nlp, batch_analyze_nlp, get_nlp_stats
from .performance_monitor import monitor, monitor_request
from .cache_manager import cache, initialize_cache, cleanup_cache
from .async_processor import processor, initialize_processor, cleanup_processor
from .load_balancer import load_balancer, initialize_load_balancer, cleanup_load_balancer
from .cdn_manager import cdn_manager, initialize_cdn_manager, cleanup_cdn_manager
import time
from collections import defaultdict
import asyncio
from functools import lru_cache
import aiofiles
from pathlib import Path as PathLib
from typing import Any, List, Dict, Optional
# External dependencies:
#   fastapi
#   pydantic
#   time
#   logging
#   collections (defaultdict)
# Internal dependencies:
#   .models (OSContentUGCVideoRequest, OSContentUGCVideoResponse, NLPRequest, NLPResponse)
#   .nlp_utils (analyze_nlp)
#   .performance_monitor (monitor)
#   .cache_manager (cache, initialize_cache)
#   .async_processor (processor, initialize_processor)
#   .load_balancer (load_balancer, initialize_load_balancer)
#   .cdn_manager (cdn_manager, initialize_cdn_manager)


logger = logging.getLogger("os_content.api")

app = FastAPI()

# Optimized rate limiter with cleanup
RATE_LIMIT = 60  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds
rate_limit_store = defaultdict(list)

def cleanup_rate_limit_store():
    """Clean up old rate limit entries"""
    current_time = time.time()
    for ip in list(rate_limit_store.keys()):
        rate_limit_store[ip] = [
            timestamp for timestamp in rate_limit_store[ip] 
            if current_time - timestamp < RATE_LIMIT_WINDOW
        ]
        if not rate_limit_store[ip]:
            del rate_limit_store[ip]

def check_rate_limit(client_ip: str) -> bool:
    """Check if client is within rate limits"""
    cleanup_rate_limit_store()
    current_time = time.time()
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT:
        return False
    rate_limit_store[client_ip].append(current_time)
    return True

# Optimized middleware with better error handling
@app.middleware("http")
async def optimized_middleware(request: Request, call_next):
    
    """optimized_middleware function."""
start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    # Rate limiting
    if not check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429, 
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Record performance metrics
        monitor.record_request(process_time, success=True)
        
        # Add performance headers
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        response.headers["X-Rate-Limit-Remaining"] = str(RATE_LIMIT - len(rate_limit_store[client_ip]))
        
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
        
    except HTTPException as exc:
        process_time = time.time() - start_time
        monitor.record_request(process_time, success=False)
        logger.warning(
            f"HTTPException: {exc.detail}",
            extra={
                "module": __name__,
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code,
                "client_ip": client_ip
            }
        )
        raise
    except Exception as exc:
        process_time = time.time() - start_time
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
            content={"detail": "Internal server error"}
        )

# Global state management
langchain_service = None
ugc_video_status_store = {}

# Optimized file operations
UPLOAD_DIR = PathLib("uploads/os_content")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=128)
def get_file_extension(filename: str) -> str:
    """Get file extension with caching"""
    return PathLib(filename).suffix.lower()

async async def save_upload_file_async(upload_file: UploadFile, folder: PathLib) -> str:
    """Async file upload with better error handling"""
    try:
        ext = get_file_extension(upload_file.filename)
        file_id = f"{uuid4()}{ext}"
        file_path = folder / file_id
        
        async with aiofiles.open(file_path, "wb") as buffer:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = await upload_file.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            await buffer.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving file {upload_file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Error saving uploaded file")

async def process_ugc_video_task(request: OSContentUGCVideoRequest, output_dir: PathLib):
    """Optimized video processing task with CDN integration"""
    try:
        output_path = output_dir / f"ugc_{request.id}.mp4"
        
        # Update status to processing
        ugc_video_status_store[request.id] = {
            "status": "processing",
            "video_url": "",
            "details": {"message": "Procesando video..."},
            "created_at": request.created_at,
            "progress": 0.1,
            "estimated_duration": request.estimated_duration,
        }
        
        # Calculate estimated duration if not provided
        if not request.estimated_duration:
            num_media = len(request.image_urls) + len(request.video_urls)
            estimated_duration = num_media * 3.0
            request.estimated_duration = estimated_duration
            ugc_video_status_store[request.id]["estimated_duration"] = estimated_duration
        
        # Process video using async processor
        video_path = await create_ugc_video_ad_with_langchain(
            image_paths=request.image_urls,
            video_paths=request.video_urls,
            text_prompt=request.text_prompt,
            output_path=str(output_path),
            langchain_service=langchain_service,
            duration_per_image=3.0,
            resolution=(1080, 1920),
            audio_path=None,
            language=request.language,
        )
        
        # Upload to CDN for optimized delivery
        cdn_url = await cdn_manager.upload_content(
            content_path=video_path,
            content_id=request.id,
            content_type="video"
        )
        
        # Update status to completed with CDN URL
        ugc_video_status_store[request.id] = {
            "status": "completed",
            "video_url": cdn_url,
            "local_path": video_path,
            "details": {"message": "Video generado exitosamente."},
            "created_at": request.created_at,
            "progress": 1.0,
            "estimated_duration": request.estimated_duration,
        }
        
        logger.info(f"Video generado exitosamente para request_id={request.id}")
        return cdn_url
        
    except Exception as e:
        logger.error(f"Error generando video para request_id={request.id}: {str(e)}")
        ugc_video_status_store[request.id] = {
            "status": "failed",
            "video_url": "",
            "details": {"message": f"Error: {str(e)}"},
            "created_at": request.created_at,
            "progress": 0.0,
            "estimated_duration": getattr(request, 'estimated_duration', None),
        }
        return None

router = APIRouter(prefix="/os-content", tags=["OS Content UGC Video"])

@router.on_event('startup')
async def on_startup() -> None:
    """Initialize resources on startup"""
    global langchain_service
    
    # Initialize cache system
    await initialize_cache()
    logger.info("Cache system initialized")
    
    # Initialize async processor
    await initialize_processor(max_concurrent=20)
    logger.info("Async processor initialized")
    
    # Initialize load balancer if backend servers configured
    backend_servers = os.getenv("BACKEND_SERVERS", "").split(",")
    if backend_servers and backend_servers[0]:
        await initialize_load_balancer(backend_servers, algorithm="round_robin")
        logger.info("Load balancer initialized")
    
    # Initialize CDN manager
    cdn_url = os.getenv("CDN_URL", "")
    await initialize_cdn_manager(cdn_url=cdn_url)
    logger.info("CDN manager initialized")
    
    # Initialize langchain service if needed
    # langchain_service = initialize_langchain_service()
    logger.info("OS Content API started successfully")

@router.on_event('shutdown')
async def on_shutdown() -> None:
    """Cleanup resources on shutdown"""
    # Cleanup temporary files
    for temp_file in UPLOAD_DIR.glob("temp_*"):
        try:
            temp_file.unlink()
        except Exception as e:
            logger.warning(f"Could not delete temp file {temp_file}: {e}")
    
    # Cleanup all systems
    await cleanup_cache()
    await cleanup_processor()
    await cleanup_load_balancer()
    await cleanup_cdn_manager()
    
    logger.info("OS Content API shutdown complete")

@router.post("/ugc-video", response_model=OSContentUGCVideoResponse, status_code=status.HTTP_202_ACCEPTED)
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
    language: Optional[str] = Form("es", description="Idioma principal del texto"),
):
    """Crea un nuevo video UGC y realiza análisis NLP del texto proporcionado."""
    
    # Validate input
    if not text_prompt.strip():
        raise HTTPException(status_code=400, detail="El texto del prompt no puede estar vacío")
    
    # Process uploaded files
    image_urls = []
    video_urls = []
    
    if images:
        for img in images:
            if img.content_type and not img.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"Archivo {img.filename} no es una imagen válida")
            path = await save_upload_file_async(img, UPLOAD_DIR)
            image_urls.append(path)
    
    if videos:
        for vid in videos:
            if vid.content_type and not vid.content_type.startswith('video/'):
                raise HTTPException(status_code=400, detail=f"Archivo {vid.filename} no es un video válido")
            path = await save_upload_file_async(vid, UPLOAD_DIR)
            video_urls.append(path)
    
    if not image_urls and not video_urls:
        raise HTTPException(status_code=400, detail="Debes subir al menos una imagen o video.")
    
    # Generate request ID and calculate duration
    request_id = str(uuid4())
    num_media = len(image_urls) + len(video_urls)
    estimated_duration = num_media * 3.0 if num_media > 0 else None
    
    # Perform NLP analysis asynchronously with caching
    try:
        nlp_analysis = await analyze_nlp(text_prompt, lang=language)
    except Exception as e:
        logger.error(f"Error en análisis NLP: {e}")
        nlp_analysis = {'error': str(e)}
    
    # Create request object
    request = OSContentUGCVideoRequest(
        id=request_id,
        user_id=user_id,
        title=title,
        description=description,
        text_prompt=text_prompt,
        image_urls=image_urls,
        video_urls=video_urls,
        ugc_type=ugc_type,
        created_at=datetime.utcnow(),
        metadata={},
        language=language,
        estimated_duration=estimated_duration,
        nlp_analysis=nlp_analysis,
    )
    
    # Add background task using async processor
    background_tasks.add_task(process_ugc_video_task, request, UPLOAD_DIR)
    
    # Create response
    response = OSContentUGCVideoResponse(
        request_id=request_id,
        video_url="",
        status="queued",
        created_at=datetime.utcnow(),
        details={"message": "Video ad UGC en cola para procesamiento automático."},
        progress=0.0,
        estimated_duration=estimated_duration,
        nlp_analysis=nlp_analysis,
    )
    
    logger.info(f"Solicitud recibida para request_id={request_id}")
    return response

@router.get("/ugc-video/{request_id}/status", response_model=OSContentUGCVideoResponse)
@monitor_request
async def get_ugc_video_status(request_id: str = Path(..., description="ID de la solicitud de video UGC")):
    """Get the status of a UGC video request"""
    status_data = ugc_video_status_store.get(request_id)
    if not status_data:
        raise HTTPException(status_code=404, detail="Request ID no encontrado")
    
    return OSContentUGCVideoResponse(
        request_id=request_id,
        video_url=status_data["video_url"],
        status=status_data["status"],
        created_at=status_data["created_at"],
        details=status_data["details"],
        progress=status_data.get("progress", None),
        estimated_duration=status_data.get("estimated_duration", None),
    )

@router.post("/nlp", response_model=NLPResponse)
@monitor_request
async def nlp_endpoint(request: NLPRequest) -> NLPResponse:
    """Analyze text using NLP"""
    try:
        result = await analyze_nlp(request.text, request.lang)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return NLPResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in NLP endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/nlp/batch")
@monitor_request
async def batch_nlp_endpoint(texts: List[str], lang: str = "es") -> List[Dict[str, Any]]:
    """Batch NLP analysis for multiple texts"""
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")
    
    try:
        results = await batch_analyze_nlp(texts, lang)
        return results
    except Exception as e:
        logger.error(f"Batch NLP error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

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
async def get_nlp_stats_endpoint():
    """Get NLP system statistics"""
    return await get_nlp_stats()

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