"""
Enhanced Video Processing API with Opus Clip Features

Integrates Content Curation Engine, Speaker Tracking System, and B-roll Integration
to create a comprehensive video processing platform like Opus Clip.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import structlog
import time
import asyncio
import uuid
from pydantic import BaseModel, field_validator
from pathlib import Path
import json
import os

# Import processors
from .processors.content_curation_engine import ContentCurationEngine, EngagementAnalyzer, SegmentDetector, ClipOptimizer
from .processors.speaker_tracking_system import SpeakerTrackingSystem, FaceDetector, ObjectTracker, AutoFramer, TrackingConfig
from .processors.broll_integration_system import BrollIntegrationSystem, ContentAnalyzer, BrollSuggester, BrollIntegrator, BrollConfig
from .processors.video_processor import VideoProcessor, VideoProcessorConfig
from .processors.viral_processor import ViralVideoProcessor, ViralProcessorConfig
from .processors.langchain_processor import LangChainVideoProcessor, LangChainConfig

# Import models
from .models.video_models import VideoClipRequest, VideoClipResponse, VideoClipBatchRequest, VideoClipBatchResponse
from .models.viral_models import ViralVideoVariant, ViralVideoBatchResponse, ViralCaptionConfig

# Import error handling
from .error_handling import ErrorHandler, ErrorCode, ValidationError, ProcessingError, ExternalServiceError

logger = structlog.get_logger("enhanced_api")
error_handler = ErrorHandler()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class OpusClipRequest(BaseModel):
    """Request model for Opus Clip processing."""
    video_path: str
    content_text: Optional[str] = None
    target_platform: str = "tiktok"  # tiktok, youtube, instagram, twitter
    enable_content_curation: bool = True
    enable_speaker_tracking: bool = True
    enable_broll_integration: bool = True
    max_clips: int = 5
    clip_duration_range: tuple = (8, 15)  # min, max seconds
    output_resolution: tuple = (1080, 1920)  # width, height
    quality: str = "high"  # low, medium, high
    
    @field_validator('target_platform')
    def validate_platform(cls, v):
        allowed = ["tiktok", "youtube", "instagram", "twitter", "facebook"]
        if v not in allowed:
            raise ValueError(f"Platform must be one of {allowed}")
        return v
    
    @field_validator('quality')
    def validate_quality(cls, v):
        allowed = ["low", "medium", "high"]
        if v not in allowed:
            raise ValueError(f"Quality must be one of {allowed}")
        return v

class OpusClipResponse(BaseModel):
    """Response model for Opus Clip processing."""
    job_id: str
    status: str
    clips: List[Dict[str, Any]]
    processing_time: float
    total_clips: int
    success_rate: float
    metadata: Dict[str, Any]

class ContentCurationRequest(BaseModel):
    """Request model for content curation."""
    video_path: str
    analysis_depth: str = "medium"  # low, medium, high
    target_duration: float = 12.0
    engagement_threshold: float = 0.7

class SpeakerTrackingRequest(BaseModel):
    """Request model for speaker tracking."""
    video_path: str
    target_resolution: tuple = (1080, 1920)
    tracking_quality: str = "high"
    enable_auto_framing: bool = True

class BrollIntegrationRequest(BaseModel):
    """Request model for B-roll integration."""
    video_path: str
    content_text: str
    broll_types: List[str] = ["stock_footage", "ai_generated", "graphics"]
    max_suggestions: int = 3

# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Opus Clip Enhanced API",
    description="AI-powered video processing with content curation, speaker tracking, and B-roll integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# GLOBAL VARIABLES AND CONFIGURATION
# =============================================================================

# Initialize processors
content_curation_engine = ContentCurationEngine()
speaker_tracking_system = SpeakerTrackingSystem()
broll_integration_system = BrollIntegrationSystem()

# Job tracking
active_jobs: Dict[str, Dict[str, Any]] = {}
completed_jobs: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracking."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "features": {
            "content_curation": True,
            "speaker_tracking": True,
            "broll_integration": True,
            "viral_optimization": True
        }
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status."""
    try:
        # Check each processor
        health_status = {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "components": {
                "content_curation_engine": "healthy",
                "speaker_tracking_system": "healthy",
                "broll_integration_system": "healthy",
                "video_processor": "healthy",
                "viral_processor": "healthy"
            },
            "active_jobs": len(active_jobs),
            "completed_jobs": len(completed_jobs)
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

# =============================================================================
# MAIN OPUS CLIP PROCESSING ENDPOINT
# =============================================================================

@app.post("/opus-clip/process", response_model=OpusClipResponse)
async def process_video_opus_clip(
    request: OpusClipRequest,
    background_tasks: BackgroundTasks
):
    """
    Process video with full Opus Clip functionality.
    
    This endpoint combines:
    - Content curation to find engaging moments
    - Speaker tracking for professional framing
    - B-roll integration for enhanced visuals
    - Platform-specific optimization
    """
    try:
        job_id = str(uuid.uuid4())
        logger.info(f"Starting Opus Clip processing job: {job_id}")
        
        # Initialize job tracking
        active_jobs[job_id] = {
            "status": "processing",
            "start_time": time.time(),
            "request": request.dict(),
            "progress": 0
        }
        
        # Start background processing
        background_tasks.add_task(
            process_opus_clip_background,
            job_id,
            request
        )
        
        return OpusClipResponse(
            job_id=job_id,
            status="processing",
            clips=[],
            processing_time=0.0,
            total_clips=0,
            success_rate=0.0,
            metadata={"message": "Processing started"}
        )
        
    except Exception as e:
        logger.error(f"Opus Clip processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

async def process_opus_clip_background(job_id: str, request: OpusClipRequest):
    """Background task for Opus Clip processing."""
    try:
        start_time = time.time()
        clips = []
        
        # Step 1: Content Curation (if enabled)
        if request.enable_content_curation:
            logger.info(f"Job {job_id}: Starting content curation")
            active_jobs[job_id]["progress"] = 10
            
            curation_result = await content_curation_engine.analyze_video(request.video_path)
            
            # Extract clips from curation result
            for clip_data in curation_result.get("clips", []):
                clips.append({
                    "start_time": clip_data["start_time"],
                    "end_time": clip_data["end_time"],
                    "duration": clip_data["duration"],
                    "score": clip_data["score"],
                    "type": "content_curated",
                    "metadata": clip_data.get("metadata", {})
                })
            
            logger.info(f"Job {job_id}: Content curation completed, found {len(clips)} clips")
        
        # Step 2: Speaker Tracking (if enabled)
        if request.enable_speaker_tracking and clips:
            logger.info(f"Job {job_id}: Starting speaker tracking")
            active_jobs[job_id]["progress"] = 50
            
            # Process each clip with speaker tracking
            tracked_clips = []
            for i, clip in enumerate(clips):
                try:
                    # Extract clip video
                    clip_path = await extract_video_clip(
                        request.video_path,
                        clip["start_time"],
                        clip["end_time"],
                        f"/tmp/clip_{job_id}_{i}.mp4"
                    )
                    
                    # Apply speaker tracking
                    tracking_result = await speaker_tracking_system.process_video(
                        clip_path,
                        f"/tmp/tracked_clip_{job_id}_{i}.mp4",
                        request.output_resolution[0],
                        request.output_resolution[1]
                    )
                    
                    # Update clip with tracking info
                    clip["tracked_video"] = f"/tmp/tracked_clip_{job_id}_{i}.mp4"
                    clip["tracking_report"] = tracking_result.get("tracking_report", {})
                    tracked_clips.append(clip)
                    
                except Exception as e:
                    logger.error(f"Speaker tracking failed for clip {i}: {e}")
                    tracked_clips.append(clip)  # Keep original clip
            
            clips = tracked_clips
            logger.info(f"Job {job_id}: Speaker tracking completed")
        
        # Step 3: B-roll Integration (if enabled)
        if request.enable_broll_integration and clips and request.content_text:
            logger.info(f"Job {job_id}: Starting B-roll integration")
            active_jobs[job_id]["progress"] = 80
            
            # Process each clip with B-roll integration
            enhanced_clips = []
            for i, clip in enumerate(clips):
                try:
                    if "tracked_video" in clip:
                        input_video = clip["tracked_video"]
                    else:
                        input_video = await extract_video_clip(
                            request.video_path,
                            clip["start_time"],
                            clip["end_time"],
                            f"/tmp/clip_{job_id}_{i}.mp4"
                        )
                    
                    # Apply B-roll integration
                    broll_result = await broll_integration_system.process_video(
                        input_video,
                        request.content_text,
                        f"/tmp/broll_clip_{job_id}_{i}.mp4"
                    )
                    
                    # Update clip with B-roll info
                    clip["broll_video"] = f"/tmp/broll_clip_{job_id}_{i}.mp4"
                    clip["broll_report"] = broll_result.get("integration_report", {})
                    enhanced_clips.append(clip)
                    
                except Exception as e:
                    logger.error(f"B-roll integration failed for clip {i}: {e}")
                    enhanced_clips.append(clip)  # Keep original clip
            
            clips = enhanced_clips
            logger.info(f"Job {job_id}: B-roll integration completed")
        
        # Step 4: Platform-specific optimization
        logger.info(f"Job {job_id}: Applying platform optimization")
        active_jobs[job_id]["progress"] = 90
        
        optimized_clips = await optimize_for_platform(clips, request.target_platform)
        
        # Step 5: Finalize results
        processing_time = time.time() - start_time
        success_rate = len(optimized_clips) / max(len(clips), 1)
        
        # Update job status
        active_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "clips": optimized_clips,
            "processing_time": processing_time,
            "success_rate": success_rate
        })
        
        # Move to completed jobs
        completed_jobs[job_id] = active_jobs.pop(job_id)
        
        logger.info(f"Job {job_id}: Processing completed in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Background processing failed for job {job_id}: {e}")
        active_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "progress": 0
        })

# =============================================================================
# INDIVIDUAL FEATURE ENDPOINTS
# =============================================================================

@app.post("/content-curation/analyze")
async def analyze_content_curation(request: ContentCurationRequest):
    """Analyze video for content curation."""
    try:
        result = await content_curation_engine.analyze_video(request.video_path)
        return {
            "status": "success",
            "analysis": result,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Content curation analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

@app.post("/speaker-tracking/track")
async def track_speaker(request: SpeakerTrackingRequest):
    """Track speaker in video."""
    try:
        result = await speaker_tracking_system.process_video(
            request.video_path,
            f"/tmp/tracked_{uuid.uuid4()}.mp4",
            request.target_resolution[0],
            request.target_resolution[1]
        )
        return {
            "status": "success",
            "tracking_result": result,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Speaker tracking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tracking failed: {e}")

@app.post("/broll-integration/integrate")
async def integrate_broll(request: BrollIntegrationRequest):
    """Integrate B-roll content into video."""
    try:
        result = await broll_integration_system.process_video(
            request.video_path,
            request.content_text,
            f"/tmp/broll_{uuid.uuid4()}.mp4"
        )
        return {
            "status": "success",
            "integration_result": result,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"B-roll integration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Integration failed: {e}")

# =============================================================================
# JOB MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a processing job."""
    if job_id in active_jobs:
        return active_jobs[job_id]
    elif job_id in completed_jobs:
        return completed_jobs[job_id]
    else:
        raise HTTPException(status_code=404, detail="Job not found")

@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return {
        "active_jobs": list(active_jobs.keys()),
        "completed_jobs": list(completed_jobs.keys()),
        "total_active": len(active_jobs),
        "total_completed": len(completed_jobs)
    }

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a processing job."""
    if job_id in active_jobs:
        active_jobs[job_id]["status"] = "cancelled"
        return {"message": "Job cancelled"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def extract_video_clip(video_path: str, start_time: float, end_time: float, output_path: str) -> str:
    """Extract a clip from video."""
    try:
        import subprocess
        
        # Use ffmpeg to extract clip
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(start_time),
            "-t", str(end_time - start_time),
            "-c", "copy",
            output_path,
            "-y"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise ProcessingError(f"Video extraction failed: {result.stderr}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Video clip extraction failed: {e}")
        raise ProcessingError(f"Video clip extraction failed: {e}")

async def optimize_for_platform(clips: List[Dict], platform: str) -> List[Dict]:
    """Optimize clips for specific platform."""
    try:
        platform_configs = {
            "tiktok": {
                "aspect_ratio": (9, 16),
                "max_duration": 15,
                "min_duration": 8,
                "resolution": (1080, 1920)
            },
            "youtube": {
                "aspect_ratio": (16, 9),
                "max_duration": 60,
                "min_duration": 10,
                "resolution": (1920, 1080)
            },
            "instagram": {
                "aspect_ratio": (1, 1),
                "max_duration": 30,
                "min_duration": 5,
                "resolution": (1080, 1080)
            },
            "twitter": {
                "aspect_ratio": (16, 9),
                "max_duration": 30,
                "min_duration": 5,
                "resolution": (1280, 720)
            }
        }
        
        config = platform_configs.get(platform, platform_configs["tiktok"])
        
        optimized_clips = []
        for clip in clips:
            # Filter by duration
            if config["min_duration"] <= clip["duration"] <= config["max_duration"]:
                clip["platform_optimized"] = True
                clip["target_platform"] = platform
                clip["target_resolution"] = config["resolution"]
                optimized_clips.append(clip)
        
        return optimized_clips
        
    except Exception as e:
        logger.error(f"Platform optimization failed: {e}")
        return clips

# =============================================================================
# FILE SERVING ENDPOINTS
# =============================================================================

@app.get("/download/{job_id}/{clip_index}")
async def download_clip(job_id: str, clip_index: int):
    """Download a processed clip."""
    try:
        if job_id not in completed_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = completed_jobs[job_id]
        clips = job.get("clips", [])
        
        if clip_index >= len(clips):
            raise HTTPException(status_code=404, detail="Clip not found")
        
        clip = clips[clip_index]
        
        # Determine which video file to serve
        video_path = None
        if "broll_video" in clip:
            video_path = clip["broll_video"]
        elif "tracked_video" in clip:
            video_path = clip["tracked_video"]
        else:
            # Extract original clip
            video_path = await extract_video_clip(
                job["request"]["video_path"],
                clip["start_time"],
                clip["end_time"],
                f"/tmp/download_{job_id}_{clip_index}.mp4"
            )
        
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"clip_{job_id}_{clip_index}.mp4"
        )
        
    except Exception as e:
        logger.error(f"Clip download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

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
# STARTUP AND SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("Opus Clip Enhanced API starting up")
    
    # Create necessary directories
    os.makedirs("/tmp", exist_ok=True)
    
    logger.info("Opus Clip Enhanced API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Opus Clip Enhanced API shutting down")
    
    # Cleanup temporary files
    import shutil
    if os.path.exists("/tmp"):
        shutil.rmtree("/tmp", ignore_errors=True)
    
    logger.info("Opus Clip Enhanced API shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


