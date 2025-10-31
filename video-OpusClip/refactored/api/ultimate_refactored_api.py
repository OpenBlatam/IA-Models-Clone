"""
Ultimate Refactored API

Complete FastAPI application integrating all refactored components including
team collaboration, scheduling automation, and advanced features.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
import asyncio
import uuid
import time
from datetime import datetime, timedelta
from pathlib import Path
import structlog
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import refactored components
from ..core.config_manager import config_manager
from ..core.job_manager import JobManager, JobPriority, JobStatus
from ..processors.refactored_content_curation import RefactoredContentCurationEngine
from ..monitoring.performance_monitor import performance_monitor
from ..testing.test_suite import TestSuite
from ..optimization.performance_optimizer import performance_optimizer
from ..collaboration.team_collaboration_system import (
    team_collaboration_system, UserRole, ProjectStatus, CollaborationEvent
)
from ..automation.scheduling_automation import (
    scheduling_automation_system, Platform, PostStatus, ScheduleType
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("ultimate_refactored_api")

# Initialize FastAPI app
app = FastAPI(
    title="Ultimate Opus Clip - Refactored API",
    description="Advanced video processing and collaboration platform with AI-powered features",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize components
job_manager = JobManager()
content_curation_engine = RefactoredContentCurationEngine()

# Pydantic models
class VideoProcessingRequest(BaseModel):
    """Request model for video processing."""
    video_path: str = Field(..., description="Path to input video file")
    features: Dict[str, bool] = Field(default_factory=dict, description="Feature toggles")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Processing settings")
    user_id: Optional[str] = Field(None, description="User ID for collaboration")
    project_id: Optional[str] = Field(None, description="Project ID for collaboration")

class VideoProcessingResponse(BaseModel):
    """Response model for video processing."""
    job_id: str
    status: str
    message: str
    estimated_completion: Optional[datetime] = None

class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class CollaborationRequest(BaseModel):
    """Request model for collaboration features."""
    project_name: str
    description: str
    video_path: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    members: List[Dict[str, str]] = Field(default_factory=list)

class SchedulingRequest(BaseModel):
    """Request model for scheduling posts."""
    video_path: str
    platforms: List[str]
    content: str
    scheduled_at: Optional[datetime] = None
    hashtags: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)

class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: datetime
    version: str
    components: Dict[str, str]
    performance: Dict[str, Any]

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track requests for performance monitoring."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Add request ID to headers
    request.state.request_id = request_id
    
    try:
        response = await call_next(request)
        
        # Record request metrics
        duration = time.time() - start_time
        performance_monitor.record_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        performance_monitor.record_request(
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration=duration
        )
        
        logger.error(f"Request failed: {e}", request_id=request_id)
        raise

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    try:
        # Start performance monitoring
        await performance_monitor.start()
        
        # Start performance optimizer
        await performance_optimizer.start()
        
        # Start scheduling automation
        await scheduling_automation_system.start()
        
        # Start team collaboration
        await team_collaboration_system.start()
        
        # Register processors
        job_manager.register_processor("content_curation", content_curation_engine.process)
        
        logger.info("Ultimate Refactored API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    try:
        # Stop all components
        await performance_monitor.stop()
        await performance_optimizer.stop()
        await scheduling_automation_system.stop()
        
        logger.info("Ultimate Refactored API stopped")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Health check endpoints
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Comprehensive health check."""
    try:
        # Get system status
        system_status = performance_monitor.get_system_status()
        
        # Check component health
        components = {
            "performance_monitor": "healthy" if performance_monitor.running else "unhealthy",
            "performance_optimizer": "healthy" if performance_optimizer.running else "unhealthy",
            "scheduling_automation": "healthy" if scheduling_automation_system.content_scheduler.running else "unhealthy",
            "team_collaboration": "healthy",
            "job_manager": "healthy"
        }
        
        # Determine overall status
        overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.now(),
            version="2.0.0",
            components=components,
            performance=system_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with performance metrics."""
    try:
        # Get performance report
        report = await performance_monitor.generate_report(timedelta(hours=1))
        
        # Get optimization status
        optimization_status = performance_optimizer.get_optimization_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "performance_report": {
                "summary": report.summary,
                "recommendations": report.recommendations
            },
            "optimization_status": optimization_status,
            "active_jobs": len(job_manager.get_active_jobs()),
            "system_metrics": performance_monitor.get_system_status()
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail="Detailed health check failed")

# Video processing endpoints
@app.post("/process/video", response_model=VideoProcessingResponse)
async def process_video(request: VideoProcessingRequest, background_tasks: BackgroundTasks):
    """Process video with Ultimate Opus Clip features."""
    try:
        # Validate video file
        if not Path(request.video_path).exists():
            raise HTTPException(status_code=400, detail="Video file not found")
        
        # Create job
        job_id = await job_manager.submit_job(
            job_type="content_curation",
            input_data={
                "video_path": request.video_path,
                "features": request.features,
                "settings": request.settings,
                "user_id": request.user_id,
                "project_id": request.project_id
            },
            priority=JobPriority.NORMAL
        )
        
        # Process job in background
        background_tasks.add_task(process_video_job, job_id)
        
        return VideoProcessingResponse(
            job_id=job_id,
            status="submitted",
            message="Video processing job submitted successfully",
            estimated_completion=datetime.now() + timedelta(minutes=5)
        )
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_video_job(job_id: str):
    """Process video job in background."""
    try:
        # Get job
        job = await job_manager.get_job(job_id)
        if not job:
            return
        
        # Process with content curation engine
        result = await content_curation_engine.process(job_id, job.input_data)
        
        # Update job with result
        await job_manager.update_job_result(job_id, result)
        
        logger.info(f"Video processing job {job_id} completed")
        
    except Exception as e:
        logger.error(f"Video processing job {job_id} failed: {e}")
        await job_manager.update_job_error(job_id, str(e))

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status and results."""
    try:
        job = await job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status.value,
            progress=job.progress,
            result=job.result_data,
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at
        )
        
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """List jobs with optional filtering."""
    try:
        jobs = job_manager.get_jobs(status=JobStatus(status) if status else None, limit=limit)
        
        return {
            "jobs": [
                {
                    "job_id": job.job_id,
                    "job_type": job.job_type,
                    "status": job.status.value,
                    "progress": job.progress,
                    "created_at": job.created_at.isoformat(),
                    "updated_at": job.updated_at.isoformat()
                }
                for job in jobs
            ],
            "total": len(jobs)
        }
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Collaboration endpoints
@app.post("/collaboration/projects")
async def create_project(request: CollaborationRequest, user_id: str = "default_user"):
    """Create a new collaboration project."""
    try:
        project = await team_collaboration_system.create_project(
            name=request.project_name,
            description=request.description,
            owner_id=user_id,
            video_path=request.video_path,
            tags=request.tags
        )
        
        # Add members if specified
        for member in request.members:
            await team_collaboration_system.add_project_member(
                project_id=project.project_id,
                user_id=member["user_id"],
                role=UserRole(member.get("role", "viewer"))
            )
        
        return {
            "project_id": project.project_id,
            "name": project.name,
            "status": "created",
            "message": "Project created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collaboration/projects/{project_id}")
async def get_project(project_id: str):
    """Get project details."""
    try:
        project = await team_collaboration_system.project_manager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get collaborators
        collaborators = await team_collaboration_system.get_project_collaborators(project_id)
        
        # Get recent activity
        activity = await team_collaboration_system.get_project_activity(project_id)
        
        return {
            "project": {
                "project_id": project.project_id,
                "name": project.name,
                "description": project.description,
                "status": project.status.value,
                "created_at": project.created_at.isoformat(),
                "updated_at": project.updated_at.isoformat(),
                "tags": project.tags
            },
            "collaborators": [
                {
                    "user_id": user.user_id,
                    "username": user.username,
                    "display_name": user.display_name,
                    "is_online": user.is_online
                }
                for user in collaborators
            ],
            "activity": activity
        }
        
    except Exception as e:
        logger.error(f"Failed to get project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collaboration/projects/{project_id}/comments")
async def add_comment(
    project_id: str,
    user_id: str,
    content: str,
    element_type: str,
    element_id: str,
    timestamp: float,
    position: Dict[str, float],
    mentions: List[str] = None
):
    """Add a comment to a project element."""
    try:
        comment = await team_collaboration_system.add_comment(
            project_id=project_id,
            user_id=user_id,
            content=content,
            element_type=element_type,
            element_id=element_id,
            timestamp=timestamp,
            position=position,
            mentions=mentions
        )
        
        return {
            "comment_id": comment.comment_id,
            "status": "created",
            "message": "Comment added successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to add comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Scheduling endpoints
@app.post("/scheduling/schedule")
async def schedule_post(request: SchedulingRequest):
    """Schedule a video post across multiple platforms."""
    try:
        # Convert platform strings to Platform enums
        platforms = [Platform(platform) for platform in request.platforms]
        
        # Schedule posts
        post_ids = await scheduling_automation_system.schedule_video_post(
            video_path=request.video_path,
            platforms=platforms,
            content=request.content,
            scheduled_at=request.scheduled_at,
            hashtags=request.hashtags,
            mentions=request.mentions
        )
        
        return {
            "post_ids": post_ids,
            "status": "scheduled",
            "message": f"Video scheduled for {len(platforms)} platforms"
        }
        
    except Exception as e:
        logger.error(f"Failed to schedule post: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scheduling/calendar")
async def get_publishing_calendar(
    start_date: datetime,
    end_date: datetime
):
    """Get publishing calendar for date range."""
    try:
        calendar = await scheduling_automation_system.get_publishing_calendar(
            start_date, end_date
        )
        
        return {
            "calendar": calendar,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get publishing calendar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scheduling/posts")
async def get_scheduled_posts(platform: Optional[str] = None, status: Optional[str] = None):
    """Get scheduled posts with optional filtering."""
    try:
        posts = await scheduling_automation_system.content_scheduler.get_scheduled_posts()
        
        # Filter by platform if specified
        if platform:
            posts = [p for p in posts if p.platform.value == platform]
        
        # Filter by status if specified
        if status:
            posts = [p for p in posts if p.status.value == status]
        
        return {
            "posts": [
                {
                    "post_id": post.post_id,
                    "platform": post.platform.value,
                    "content": post.content,
                    "scheduled_at": post.scheduled_at.isoformat(),
                    "status": post.status.value,
                    "created_at": post.created_at.isoformat()
                }
                for post in posts
            ],
            "total": len(posts)
        }
        
    except Exception as e:
        logger.error(f"Failed to get scheduled posts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Performance and monitoring endpoints
@app.get("/performance/metrics")
async def get_performance_metrics():
    """Get current performance metrics."""
    try:
        system_status = performance_monitor.get_system_status()
        optimization_status = performance_optimizer.get_optimization_status()
        
        return {
            "system_status": system_status,
            "optimization_status": optimization_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/report")
async def get_performance_report(hours: int = 1):
    """Get performance report for specified time range."""
    try:
        report = await performance_monitor.generate_report(timedelta(hours=hours))
        
        return {
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "time_range_hours": hours,
            "summary": report.summary,
            "recommendations": report.recommendations,
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in report.alerts
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Testing endpoints
@app.post("/testing/run-tests")
async def run_tests():
    """Run comprehensive test suite."""
    try:
        test_suite = TestSuite()
        results = await test_suite.run_all_tests()
        
        # Calculate summary
        total_tests = len(results)
        passed_tests = len([r for r in results if r.success])
        failed_tests = total_tests - passed_tests
        
        return {
            "status": "completed",
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "results": [
                {
                    "test_name": result.test_name,
                    "test_type": result.test_type.value,
                    "success": result.success,
                    "duration": result.duration,
                    "error_message": result.error_message
                }
                for result in results
            ]
        }
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoints
@app.get("/config")
async def get_configuration():
    """Get current system configuration."""
    try:
        return {
            "features": {
                "content_curation": config_manager.is_feature_enabled("content_curation"),
                "speaker_tracking": config_manager.is_feature_enabled("speaker_tracking"),
                "broll_integration": config_manager.is_feature_enabled("broll_integration"),
                "viral_scoring": config_manager.is_feature_enabled("viral_scoring"),
                "audio_processing": config_manager.is_feature_enabled("audio_processing"),
                "export_integration": config_manager.is_feature_enabled("export_integration"),
                "team_collaboration": config_manager.is_feature_enabled("team_collaboration"),
                "scheduling_automation": config_manager.is_feature_enabled("scheduling_automation")
            },
            "settings": {
                "max_workers": config_manager.get("processing.max_workers", 4),
                "timeout": config_manager.get("processing.timeout", 300),
                "retry_attempts": config_manager.get("processing.retry_attempts", 3)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/config")
async def update_configuration(config: Dict[str, Any]):
    """Update system configuration."""
    try:
        for key, value in config.items():
            config_manager.set(key, value)
        
        return {
            "status": "updated",
            "message": "Configuration updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", request_id=getattr(request.state, "request_id", None))
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "ultimate_refactored_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


