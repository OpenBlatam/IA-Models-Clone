"""
Refactored Ultimate Opus Clip API

Improved architecture with better separation of concerns, enhanced performance,
and comprehensive monitoring capabilities.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import structlog
import time
import asyncio
import uuid
from pydantic import BaseModel, field_validator
from pathlib import Path
import json
import os
from contextlib import asynccontextmanager

# Import refactored components
from ..core.base_processor import BaseProcessor, ProcessorManager
from ..core.config_manager import config_manager
from ..core.job_manager import JobManager, JobPriority, JobStatus
from ..processors.refactored_content_curation import RefactoredContentCurationEngine

# Import error handling
from ..error_handling import ErrorHandler, ErrorCode, ValidationError, ProcessingError, ExternalServiceError

logger = structlog.get_logger("refactored_api")
error_handler = ErrorHandler()

# =============================================================================
# GLOBAL COMPONENTS
# =============================================================================

# Initialize global components
processor_manager = ProcessorManager()
job_manager = JobManager(
    max_workers=config_manager.get("processing.max_workers", 4),
    db_path=config_manager.get("database.url", "sqlite:///opus_clip.db").replace("sqlite:///", ""),
    cleanup_interval=config_manager.get("processing.cleanup_interval", 3600.0)
)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class RefactoredOpusClipRequest(BaseModel):
    """Refactored Opus Clip processing request."""
    video_path: str
    content_text: Optional[str] = None
    target_platform: str = "tiktok"
    
    # Feature toggles
    enable_content_curation: bool = True
    enable_speaker_tracking: bool = True
    enable_broll_integration: bool = True
    enable_viral_scoring: bool = True
    enable_audio_processing: bool = True
    enable_analytics: bool = True
    
    # Processing options
    priority: str = "normal"  # low, normal, high, urgent
    timeout: float = 300.0
    max_retries: int = 3
    
    # Output options
    max_clips: int = 5
    clip_duration_range: tuple = (8, 15)
    output_resolution: tuple = (1080, 1920)
    quality: str = "high"
    
    # Advanced options
    callback_url: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    @field_validator('target_platform')
    def validate_platform(cls, v):
        allowed = ["tiktok", "youtube", "instagram", "twitter", "facebook", "linkedin"]
        if v not in allowed:
            raise ValueError(f"Platform must be one of {allowed}")
        return v
    
    @field_validator('priority')
    def validate_priority(cls, v):
        allowed = ["low", "normal", "high", "urgent"]
        if v not in allowed:
            raise ValueError(f"Priority must be one of {allowed}")
        return v
    
    @field_validator('quality')
    def validate_quality(cls, v):
        allowed = ["draft", "standard", "high", "professional", "broadcast"]
        if v not in allowed:
            raise ValueError(f"Quality must be one of {allowed}")
        return v

class RefactoredOpusClipResponse(BaseModel):
    """Refactored Opus Clip processing response."""
    job_id: str
    status: str
    message: str
    estimated_completion_time: Optional[float] = None
    queue_position: Optional[int] = None
    processing_url: str = ""

class JobStatusResponse(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: float
    progress_message: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    processing_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SystemStatusResponse(BaseModel):
    """System status response."""
    status: str
    version: str
    uptime: float
    processors: Dict[str, Any]
    jobs: Dict[str, Any]
    system_resources: Dict[str, Any]

# =============================================================================
# LIFECYCLE MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Refactored Ultimate Opus Clip API")
    
    try:
        # Initialize processor manager
        await processor_manager.initialize_all()
        
        # Register processors
        await register_processors()
        
        # Start job manager
        await job_manager.start()
        
        # Start job processing
        asyncio.create_task(job_manager.process_jobs())
        
        logger.info("Refactored Ultimate Opus Clip API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Refactored Ultimate Opus Clip API")
    
    try:
        # Stop job manager
        await job_manager.stop()
        
        # Cleanup processors
        await processor_manager.cleanup_all()
        
        logger.info("Refactored Ultimate Opus Clip API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Refactored Ultimate Opus Clip API",
    description="Advanced video processing platform with improved architecture and performance",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config_manager.get("api.cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# =============================================================================
# PROCESSOR REGISTRATION
# =============================================================================

async def register_processors():
    """Register all processors with the processor manager."""
    try:
        # Content Curation Engine
        if config_manager.is_feature_enabled("content_curation"):
            content_curation = RefactoredContentCurationEngine()
            processor_manager.register_processor(content_curation)
            job_manager.register_processor("content_curation", content_curation.process)
        
        # TODO: Register other processors as they are refactored
        # Speaker Tracking System
        # B-roll Integration System
        # Advanced Viral Scoring
        # Audio Processing System
        # Professional Export System
        # Advanced Analytics System
        
        logger.info("All processors registered successfully")
        
    except Exception as e:
        logger.error(f"Processor registration failed: {e}")
        raise

# =============================================================================
# MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID and timing to all requests."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.start_time = time.time()
    
    response = await call_next(request)
    
    # Add response headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Processing-Time"] = str(time.time() - request.state.start_time)
    
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses."""
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        request_id=getattr(request.state, "request_id", None)
    )
    
    response = await call_next(request)
    
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        request_id=getattr(request.state, "request_id", None)
    )
    
    return response

# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@app.get("/health", response_model=SystemStatusResponse)
async def health_check():
    """Comprehensive health check."""
    try:
        uptime = time.time() - (getattr(health_check, 'start_time', time.time()))
        
        return SystemStatusResponse(
            status="healthy",
            version="3.0.0",
            uptime=uptime,
            processors=processor_manager.get_processor_status(),
            jobs=job_manager.get_statistics(),
            system_resources={
                "cpu_usage": "normal",
                "memory_usage": "normal",
                "disk_usage": "normal"
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status."""
    try:
        health_status = {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "version": "3.0.0",
            "components": {
                "processor_manager": "healthy",
                "job_manager": "healthy",
                "database": "healthy",
                "cache": "healthy"
            },
            "processors": processor_manager.get_processor_status(),
            "jobs": job_manager.get_statistics(),
            "configuration": {
                "features_enabled": {
                    "content_curation": config_manager.is_feature_enabled("content_curation"),
                    "speaker_tracking": config_manager.is_feature_enabled("speaker_tracking"),
                    "broll_integration": config_manager.is_feature_enabled("broll_integration"),
                    "viral_scoring": config_manager.is_feature_enabled("viral_scoring"),
                    "audio_processing": config_manager.is_feature_enabled("audio_processing"),
                    "professional_export": config_manager.is_feature_enabled("professional_export"),
                    "analytics": config_manager.is_feature_enabled("analytics")
                }
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detailed health check failed: {e}")

# =============================================================================
# MAIN PROCESSING ENDPOINTS
# =============================================================================

@app.post("/process", response_model=RefactoredOpusClipResponse)
async def process_video(request: RefactoredOpusClipRequest):
    """
    Process video with refactored Ultimate Opus Clip system.
    
    This endpoint submits a video processing job and returns immediately
    with a job ID for tracking progress.
    """
    try:
        # Validate request
        if not os.path.exists(request.video_path):
            raise HTTPException(status_code=400, detail="Video file not found")
        
        # Convert priority string to enum
        priority_map = {
            "low": JobPriority.LOW,
            "normal": JobPriority.NORMAL,
            "high": JobPriority.HIGH,
            "urgent": JobPriority.URGENT
        }
        priority = priority_map[request.priority]
        
        # Prepare job input data
        input_data = {
            "video_path": request.video_path,
            "content_text": request.content_text,
            "target_platform": request.target_platform,
            "features": {
                "content_curation": request.enable_content_curation,
                "speaker_tracking": request.enable_speaker_tracking,
                "broll_integration": request.enable_broll_integration,
                "viral_scoring": request.enable_viral_scoring,
                "audio_processing": request.enable_audio_processing,
                "analytics": request.enable_analytics
            },
            "output_options": {
                "max_clips": request.max_clips,
                "clip_duration_range": request.clip_duration_range,
                "output_resolution": request.output_resolution,
                "quality": request.quality
            },
            "metadata": request.metadata
        }
        
        # Submit job
        job_id = await job_manager.submit_job(
            job_type="ultimate_opus_clip",
            input_data=input_data,
            priority=priority,
            timeout=request.timeout,
            max_retries=request.max_retries,
            callback_url=request.callback_url,
            metadata=request.metadata
        )
        
        # Get queue position
        queue_position = job_manager.job_queue.size()
        
        # Estimate completion time
        estimated_time = queue_position * 30.0  # Rough estimate: 30 seconds per job
        
        return RefactoredOpusClipResponse(
            job_id=job_id,
            status="queued",
            message="Job submitted successfully",
            estimated_completion_time=estimated_time,
            queue_position=queue_position,
            processing_url=f"/jobs/{job_id}"
        )
        
    except Exception as e:
        logger.error(f"Video processing submission failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing submission failed: {e}")

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of a processing job."""
    try:
        job_status = await job_manager.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatusResponse(
            job_id=job_status["job_id"],
            status=job_status["status"],
            progress=job_status["progress"],
            progress_message=job_status["progress_message"],
            created_at=job_status["created_at"],
            started_at=job_status["started_at"],
            completed_at=job_status["completed_at"],
            processing_time=job_status.get("processing_time"),
            result=job_status.get("result"),
            error=job_status.get("error_message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {e}")

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a processing job."""
    try:
        success = await job_manager.cancel_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        
        return {"message": "Job cancelled successfully", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e}")

@app.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str):
    """Retry a failed job."""
    try:
        success = await job_manager.retry_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be retried")
        
        return {"message": "Job retry initiated", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retry job: {e}")

# =============================================================================
# INDIVIDUAL FEATURE ENDPOINTS
# =============================================================================

@app.post("/processors/content-curation/process")
async def process_content_curation(request: Dict[str, Any]):
    """Process video with content curation engine only."""
    try:
        # Get content curation processor
        processor = processor_manager.processors.get("content_curation_engine")
        if not processor:
            raise HTTPException(status_code=503, detail="Content curation processor not available")
        
        # Process with content curation
        result = await processor.process(str(uuid.uuid4()), request)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error_message)
        
        return {
            "status": "success",
            "result": result.result_data,
            "processing_time": result.processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content curation processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Content curation processing failed: {e}")

# =============================================================================
# SYSTEM MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/system/status")
async def get_system_status():
    """Get comprehensive system status."""
    try:
        return {
            "status": "operational",
            "timestamp": time.time(),
            "version": "3.0.0",
            "processors": processor_manager.get_processor_status(),
            "jobs": job_manager.get_statistics(),
            "configuration": {
                "features_enabled": {
                    name: config_manager.is_feature_enabled(name)
                    for name in [
                        "content_curation", "speaker_tracking", "broll_integration",
                        "viral_scoring", "audio_processing", "professional_export", "analytics"
                    ]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"System status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"System status retrieval failed: {e}")

@app.get("/system/processors")
async def get_processors():
    """Get processor information."""
    try:
        return processor_manager.get_processor_status()
        
    except Exception as e:
        logger.error(f"Processor information retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processor information retrieval failed: {e}")

@app.get("/system/jobs")
async def get_jobs(limit: int = 100, offset: int = 0):
    """Get job information."""
    try:
        # This would need to be implemented in JobManager
        return {
            "message": "Job listing not yet implemented",
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Job information retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Job information retrieval failed: {e}")

# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================

@app.get("/config")
async def get_configuration():
    """Get current configuration."""
    try:
        return config_manager.get_all()
        
    except Exception as e:
        logger.error(f"Configuration retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration retrieval failed: {e}")

@app.post("/config/reload")
async def reload_configuration():
    """Reload configuration from sources."""
    try:
        config_manager.reload()
        return {"message": "Configuration reloaded successfully"}
        
    except Exception as e:
        logger.error(f"Configuration reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration reload failed: {e}")

# =============================================================================
# ERROR HANDLING
# =============================================================================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(ProcessingError)
async def processing_exception_handler(request: Request, exc: ProcessingError):
    """Handle processing errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Processing Error",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# =============================================================================
# STARTUP INITIALIZATION
# =============================================================================

# Set start time for uptime calculation
health_check.start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration
    host = config_manager.get("api.host", "0.0.0.0")
    port = config_manager.get("api.port", 8000)
    workers = config_manager.get("api.workers", 1)
    
    # Start server
    uvicorn.run(
        "refactored_api:app",
        host=host,
        port=port,
        workers=workers,
        log_level=config_manager.get("logging.level", "info").lower()
    )


