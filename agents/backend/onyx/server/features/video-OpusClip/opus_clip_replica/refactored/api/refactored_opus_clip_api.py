"""
Refactored Opus Clip API

Enhanced FastAPI application with:
- Refactored architecture
- Async processing
- Job management
- Error handling
- Performance monitoring
- Caching
- Modular design
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import asyncio
import uuid
import time
from datetime import datetime
import structlog
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

from ..core.config_manager import ConfigManager, Environment
from ..core.job_manager import JobManager, JobPriority
from ..core.base_processor import ProcessorConfig
from ..processors.refactored_analyzer import RefactoredOpusClipAnalyzer
from ..processors.refactored_exporter import RefactoredOpusClipExporter

logger = structlog.get_logger("refactored_api")

# Initialize FastAPI app
app = FastAPI(
    title="Refactored Opus Clip",
    description="AI-powered video clip generator with enhanced architecture",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global instances
config_manager: Optional[ConfigManager] = None
job_manager: Optional[JobManager] = None
analyzer: Optional[RefactoredOpusClipAnalyzer] = None
exporter: Optional[RefactoredOpusClipExporter] = None

# Pydantic models
class VideoAnalysisRequest(BaseModel):
    video_path: str = Field(..., description="Path to video file")
    max_clips: int = Field(default=10, ge=1, le=50, description="Maximum number of clips")
    min_duration: float = Field(default=3.0, ge=1.0, le=60.0, description="Minimum clip duration")
    max_duration: float = Field(default=30.0, ge=1.0, le=300.0, description="Maximum clip duration")
    priority: str = Field(default="normal", regex="^(low|normal|high|urgent)$", description="Job priority")

class ClipExportRequest(BaseModel):
    video_path: str = Field(..., description="Path to video file")
    segments: List[Dict[str, Any]] = Field(..., description="Video segments to export")
    output_format: str = Field(default="mp4", regex="^(mp4|mov|avi)$", description="Output format")
    quality: str = Field(default="high", regex="^(low|medium|high|ultra)$", description="Export quality")
    output_dir: Optional[str] = Field(None, description="Output directory")
    priority: str = Field(default="normal", regex="^(low|normal|high|urgent)$", description="Job priority")

class ViralScoreRequest(BaseModel):
    content: str = Field(..., description="Content to analyze")
    platform: str = Field(default="youtube", regex="^(youtube|tiktok|instagram|facebook|twitter)$", description="Target platform")

class JobStatusRequest(BaseModel):
    job_id: str = Field(..., description="Job ID to check")

class BatchAnalysisRequest(BaseModel):
    videos: List[Dict[str, Any]] = Field(..., description="List of videos to analyze")
    options: Dict[str, Any] = Field(default_factory=dict, description="Analysis options")

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    global config_manager, job_manager, analyzer, exporter
    
    try:
        # Initialize configuration manager
        config_manager = ConfigManager()
        
        # Initialize job manager
        job_manager = JobManager(
            max_workers=config_manager.performance.max_workers,
            enable_persistence=True
        )
        
        # Initialize processors
        processor_config = ProcessorConfig(
            max_retries=3,
            timeout_seconds=300.0,
            enable_caching=config_manager.performance.enable_caching,
            cache_ttl_seconds=config_manager.performance.cache_ttl_seconds,
            enable_monitoring=config_manager.performance.enable_monitoring
        )
        
        analyzer = RefactoredOpusClipAnalyzer(processor_config, config_manager)
        exporter = RefactoredOpusClipExporter(processor_config, config_manager)
        
        # Register job processors
        job_manager.register_processor("video_analysis", analyzer.process)
        job_manager.register_processor("clip_export", exporter.process)
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config_manager.security.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Start job processing
        await job_manager._start_processing()
        
        logger.info("Refactored Opus Clip API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the application."""
    global job_manager, analyzer, exporter
    
    try:
        if job_manager:
            await job_manager.shutdown()
        
        if analyzer:
            await analyzer.shutdown()
        
        if exporter:
            await exporter.shutdown()
        
        logger.info("Refactored Opus Clip API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# API Endpoints
@app.post("/api/analyze")
async def analyze_video(request: VideoAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze video with enhanced async processing."""
    try:
        # Convert priority string to enum
        priority_map = {
            "low": JobPriority.LOW,
            "normal": JobPriority.NORMAL,
            "high": JobPriority.HIGH,
            "urgent": JobPriority.URGENT
        }
        priority = priority_map.get(request.priority, JobPriority.NORMAL)
        
        # Submit job
        job_id = await job_manager.submit_job(
            job_type="video_analysis",
            data={
                "video_path": request.video_path,
                "max_clips": request.max_clips,
                "min_duration": request.min_duration,
                "max_duration": request.max_duration
            },
            priority=priority,
            metadata={
                "endpoint": "analyze",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Video analysis job submitted successfully",
            "status_url": f"/api/job/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract")
async def extract_clips(request: ClipExportRequest, background_tasks: BackgroundTasks):
    """Extract video clips with enhanced async processing."""
    try:
        # Convert priority string to enum
        priority_map = {
            "low": JobPriority.LOW,
            "normal": JobPriority.NORMAL,
            "high": JobPriority.HIGH,
            "urgent": JobPriority.URGENT
        }
        priority = priority_map.get(request.priority, JobPriority.NORMAL)
        
        # Submit job
        job_id = await job_manager.submit_job(
            job_type="clip_export",
            data={
                "video_path": request.video_path,
                "segments": request.segments,
                "output_format": request.output_format,
                "quality": request.quality,
                "output_dir": request.output_dir
            },
            priority=priority,
            metadata={
                "endpoint": "extract",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Clip export job submitted successfully",
            "status_url": f"/api/job/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"Clip extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/viral-score")
async def calculate_viral_score(request: ViralScoreRequest):
    """Calculate viral score (synchronous for quick response)."""
    try:
        # This could be made async with job management if needed
        # For now, keep it synchronous for quick response
        
        # Simple viral score calculation
        word_count = len(request.content.split())
        
        # Platform factors
        platform_factors = {
            "youtube": 0.8,
            "tiktok": 0.9,
            "instagram": 0.7,
            "facebook": 0.6,
            "twitter": 0.5
        }
        
        # Calculate score
        base_score = 0.5
        length_factor = 0.2 if 50 <= word_count <= 200 else 0.1
        platform_factor = platform_factors.get(request.platform, 0.5) * 0.3
        
        viral_score = min(base_score + length_factor + platform_factor, 1.0)
        
        return {
            "success": True,
            "data": {
                "viral_score": viral_score,
                "content_length": word_count,
                "platform": request.platform,
                "viral_potential": _get_viral_potential_label(viral_score)
            },
            "message": "Viral score calculated successfully"
        }
        
    except Exception as e:
        logger.error(f"Viral score calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and result."""
    try:
        status = await job_manager.get_job_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get result if completed
        result = None
        if status["status"] == "completed":
            result = await job_manager.get_job_result(job_id)
        
        return {
            "success": True,
            "job": status,
            "result": result.data if result else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/job/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a job."""
    try:
        success = await job_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        
        return {
            "success": True,
            "message": f"Job {job_id} cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/job/{job_id}/retry")
async def retry_job(job_id: str):
    """Retry a failed job."""
    try:
        success = await job_manager.retry_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be retried")
        
        return {
            "success": True,
            "message": f"Job {job_id} retried successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch/analyze")
async def batch_analyze_videos(request: BatchAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze multiple videos in batch."""
    try:
        job_ids = []
        
        for video_data in request.videos:
            job_id = await job_manager.submit_job(
                job_type="video_analysis",
                data={
                    "video_path": video_data["video_path"],
                    "max_clips": video_data.get("max_clips", 10),
                    "min_duration": video_data.get("min_duration", 3.0),
                    "max_duration": video_data.get("max_duration", 30.0)
                },
                priority=JobPriority.NORMAL,
                metadata={
                    "endpoint": "batch_analyze",
                    "batch_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat()
                }
            )
            job_ids.append(job_id)
        
        return {
            "success": True,
            "job_ids": job_ids,
            "total_jobs": len(job_ids),
            "message": f"Batch analysis submitted with {len(job_ids)} jobs"
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics():
    """Get system statistics."""
    try:
        job_stats = await job_manager.get_statistics()
        analyzer_stats = await analyzer.get_status() if analyzer else {}
        exporter_stats = await exporter.get_status() if exporter else {}
        
        return {
            "success": True,
            "data": {
                "job_manager": job_stats,
                "analyzer": analyzer_stats,
                "exporter": exporter_stats,
                "config": config_manager.get_config_summary() if config_manager else {}
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "components": {
                "config_manager": config_manager is not None,
                "job_manager": job_manager is not None,
                "analyzer": analyzer is not None,
                "exporter": exporter is not None
            },
            "environment": config_manager.environment.value if config_manager else "unknown"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    try:
        if not config_manager:
            raise HTTPException(status_code=503, detail="Configuration not available")
        
        return {
            "success": True,
            "config": config_manager.get_config_summary()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/reload")
async def reload_config():
    """Reload configuration."""
    try:
        if not config_manager:
            raise HTTPException(status_code=503, detail="Configuration not available")
        
        await config_manager.reload_config()
        
        return {
            "success": True,
            "message": "Configuration reloaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reload config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Refactored Opus Clip API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "/api/analyze",
            "extract": "/api/extract",
            "viral_score": "/api/viral-score",
            "job_status": "/api/job/{job_id}",
            "batch_analyze": "/api/batch/analyze",
            "statistics": "/api/statistics",
            "health": "/api/health",
            "config": "/api/config"
        }
    }

def _get_viral_potential_label(score: float) -> str:
    """Get viral potential label."""
    if score >= 0.8:
        return "High"
    elif score >= 0.6:
        return "Medium"
    elif score >= 0.4:
        return "Low"
    else:
        return "Very Low"

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "refactored_opus_clip_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


