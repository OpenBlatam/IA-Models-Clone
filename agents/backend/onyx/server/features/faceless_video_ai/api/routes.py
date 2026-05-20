"""
API routes for Faceless Video AI
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from typing import Optional
import logging
from uuid import UUID

from ..core.models import (
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoStatus,
    GenerationProgress,
)
from ..services.video_orchestrator import VideoOrchestrator
from ..services.analytics import get_analytics_service
from ..services.webhook_service import WebhookService
from ..services.batch_processor import BatchProcessor
from ..services.templates import TemplateService
from ..services.rate_limiter import get_rate_limiter
from ..services.versioning import get_versioning_service
from ..services.platform_exporter import get_platform_exporter
from ..services.custom_templates import get_custom_template_service
from ..services.recommendations import get_recommendation_service
from ..services.music_library import get_music_library
from ..services.ab_testing import get_ab_testing_service
from ..services.collaboration import get_collaboration_service, SharePermission
from ..services.scheduler import get_scheduler_service
from ..services.watermarking import get_watermark_service
from ..services.transcription import get_transcription_service
from ..services.notifications import get_notification_service
from ..services.visual_effects import get_visual_effects_service
from ..services.admin import get_dashboard_service, get_backup_service, get_monitoring_service
from ..services.performance import get_profiler_service
from ..services.config_manager import get_config_manager
from ..services.validation import get_input_validator, get_input_sanitizer
from ..services.advanced_rate_limiter import get_advanced_rate_limiter
from ..services.metrics import get_prometheus_metrics
from ..services.alerts import get_alert_manager
from ..services.feedback import get_feedback_service
from ..services.search import get_search_service
from ..services.export import get_data_exporter
from ..services.events import get_event_bus, EventType, VideoEvent
from fastapi import HTTPException, Header, Depends, Request, Response
from fastapi.responses import FileResponse
from datetime import datetime

# Initialize orchestrator (singleton pattern)
_orchestrator: Optional[VideoOrchestrator] = None
_webhook_service: Optional[WebhookService] = None

def get_orchestrator() -> VideoOrchestrator:
    """Get orchestrator instance (singleton)"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = VideoOrchestrator()
    return _orchestrator

def get_webhook_service() -> WebhookService:
    """Get webhook service instance (singleton)"""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
    return _webhook_service

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory storage for video generation jobs (use database in production)
video_jobs = {}


@router.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Generate a faceless video from a script
    
    Args:
        request: Video generation request with script and configuration
        background_tasks: FastAPI background tasks
        x_api_key: Optional API key for authentication
        
    Returns:
        Video generation response with job ID and initial status
    """
    # Rate limiting
    rate_limiter = get_rate_limiter()
    client_id = x_api_key or "anonymous"
    allowed, rate_info = rate_limiter.is_allowed(f"generate:{client_id}")
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(rate_info["reset_at"]),
            }
        )
    
    try:
        orchestrator = get_orchestrator()
        
        # Start video generation in background
        response = await orchestrator.start_generation(request)
        
        # Store job
        video_jobs[response.video_id] = response
        
        # Process in background
        background_tasks.add_task(
            orchestrator.process_generation,
            response.video_id,
            request
        )
        
        logger.info(f"Started video generation: {response.video_id}")
        
        # Add rate limit headers
        headers = {}
        if rate_info:
            headers = {
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": str(rate_info["remaining"]),
                "X-RateLimit-Reset": str(rate_info["reset_at"]),
            }
        
        # Create response with headers (FastAPI will handle this)
        return response
        
    except Exception as e:
        logger.error(f"Error starting video generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{video_id}", response_model=VideoGenerationResponse)
async def get_video_status(video_id: UUID):
    """
    Get status of video generation
    
    Args:
        video_id: Video generation job ID
        
    Returns:
        Current status and progress
    """
    if video_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Video job not found")
    
    return video_jobs[video_id]


@router.get("/videos/{video_id}/download")
async def download_video(video_id: UUID):
    """
    Download generated video
    
    Args:
        video_id: Video generation job ID
        
    Returns:
        Video file
    """
    if video_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Video job not found")
    
    job = video_jobs[video_id]
    
    if job.status != VideoStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Video not ready. Status: {job.status}"
        )
    
    if not job.video_url:
        raise HTTPException(status_code=404, detail="Video file not found")
    
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    video_path = Path(job.video_url)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"video_{video_id}.mp4"
    )


@router.post("/upload-script")
async def upload_script(file: UploadFile = File(...)):
    """
    Upload a script file (text, markdown, etc.)
    
    Args:
        file: Script file
        
    Returns:
        Extracted script text
    """
    try:
        content = await file.read()
        text = content.decode("utf-8")
        
        return {
            "text": text,
            "filename": file.filename,
            "size": len(content),
        }
    except Exception as e:
        logger.error(f"Error uploading script: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/videos/{video_id}")
async def delete_video(video_id: UUID):
    """
    Delete a video generation job and files
    
    Args:
        video_id: Video generation job ID
    """
    if video_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Video job not found")
    
    job = video_jobs[video_id]
    
    # Delete files
    if job.video_url:
        from pathlib import Path
        video_path = Path(job.video_url)
        if video_path.exists():
            video_path.unlink()
    
    if job.thumbnail_url:
        from pathlib import Path
        thumbnail_path = Path(job.thumbnail_url)
        if thumbnail_path.exists():
            thumbnail_path.unlink()
    
    # Unregister webhooks
    webhook_service = get_webhook_service()
    webhook_service.unregister_webhook(video_id)
    
    # Remove from jobs
    del video_jobs[video_id]
    
    return {"message": "Video deleted successfully"}


@router.post("/videos/{video_id}/webhook")
async def register_webhook(video_id: UUID, webhook_url: str):
    """
    Register a webhook URL for video generation notifications
    
    Args:
        video_id: Video generation job ID
        webhook_url: URL to receive webhook notifications
    """
    if video_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Video job not found")
    
    webhook_service = get_webhook_service()
    webhook_service.register_webhook(video_id, webhook_url)
    
    return {"message": "Webhook registered successfully"}


@router.get("/analytics")
async def get_analytics():
    """
    Get analytics and metrics
    
    Returns:
        Analytics data
    """
    analytics_service = get_analytics_service()
    metrics = analytics_service.get_metrics()
    usage_stats = analytics_service.get_usage_statistics()
    top_errors = analytics_service.get_top_errors()
    
    return {
        "metrics": metrics,
        "usage_statistics": usage_stats,
        "top_errors": top_errors,
    }


@router.post("/batch/generate")
async def batch_generate(
    requests: List[VideoGenerationRequest],
    background_tasks: BackgroundTasks,
    webhook_url: Optional[str] = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Generate multiple videos in batch
    
    Args:
        requests: List of video generation requests
        background_tasks: FastAPI background tasks
        webhook_url: Optional webhook URL for batch completion
        x_api_key: Optional API key for authentication
        
    Returns:
        Batch processing result
    """
    # Rate limiting for batch
    rate_limiter = get_rate_limiter()
    client_id = x_api_key or "anonymous"
    allowed, rate_info = rate_limiter.is_allowed(f"batch:{client_id}")
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for batch processing",
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(rate_info["reset_at"]),
            }
        )
    
    if len(requests) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 videos per batch")
    
    try:
        batch_processor = BatchProcessor(max_concurrent=5)
        result = await batch_processor.process_batch(requests, webhook_url)
        return result
    except Exception as e:
        logger.error(f"Batch generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch/status")
async def get_batch_status(video_ids: str):
    """
    Get status of multiple video generation jobs
    
    Args:
        video_ids: Comma-separated list of video IDs
        
    Returns:
        Batch status information
    """
    from uuid import UUID
    
    try:
        ids = [UUID(id.strip()) for id in video_ids.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video IDs format")
    
    batch_processor = BatchProcessor()
    status = await batch_processor.get_batch_status(ids)
    return status


@router.get("/templates")
async def list_templates():
    """
    List all available video templates
    
    Returns:
        List of available templates
    """
    templates = TemplateService.list_templates()
    return {"templates": templates}


@router.get("/templates/{template_name}")
async def get_template(template_name: str):
    """
    Get template configuration
    
    Args:
        template_name: Name of the template
        
    Returns:
        Template configuration
    """
    template = TemplateService.get_template(template_name)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
    
    return template


@router.post("/templates/{template_name}/generate")
async def generate_from_template(
    template_name: str,
    script_text: str,
    language: str = "es",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Generate video using a template
    
    Args:
        template_name: Name of the template
        script_text: Script text
        language: Language code
        background_tasks: FastAPI background tasks
        x_api_key: Optional API key
        
    Returns:
        Video generation response
    """
    from ..core.models import VideoScript, VideoGenerationRequest
    
    try:
        template_config = TemplateService.apply_template(template_name, script_text, language)
        
        request = VideoGenerationRequest(
            script=VideoScript(text=script_text, language=language),
            video_config=template_config["video_config"],
            audio_config=template_config["audio_config"],
            subtitle_config=template_config["subtitle_config"],
        )
        
        # Use the regular generate endpoint logic
        orchestrator = get_orchestrator()
        response = await orchestrator.start_generation(request)
        
        video_jobs[response.video_id] = response
        
        background_tasks.add_task(
            orchestrator.process_generation,
            response.video_id,
            request
        )
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Template generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos/{video_id}/versions")
async def get_video_versions(video_id: UUID):
    """
    Get all versions of a video
    
    Args:
        video_id: Video ID
        
    Returns:
        List of versions
    """
    versioning_service = get_versioning_service()
    versions = versioning_service.get_versions(video_id)
    
    return {
        "video_id": str(video_id),
        "versions": [v.to_dict() for v in versions],
        "total_versions": len(versions),
    }


@router.get("/videos/{video_id}/versions/{version_number}")
async def get_video_version(video_id: UUID, version_number: int):
    """Get specific version of a video"""
    versioning_service = get_versioning_service()
    version = versioning_service.get_version(video_id, version_number)
    
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    
    return version.to_dict()


@router.get("/videos/{video_id}/versions/compare")
async def compare_versions(video_id: UUID, version1: int, version2: int):
    """Compare two versions of a video"""
    versioning_service = get_versioning_service()
    
    try:
        comparison = versioning_service.compare_versions(video_id, version1, version2)
        return comparison
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/videos/{video_id}/export")
async def export_video(
    video_id: UUID,
    platforms: List[str],
    background_tasks: BackgroundTasks
):
    """
    Export video for multiple platforms
    
    Args:
        video_id: Video ID
        platforms: List of platforms to export for
        background_tasks: FastAPI background tasks
        
    Returns:
        Export job information
    """
    if video_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Video job not found")
    
    job = video_jobs[video_id]
    
    if job.status != VideoStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Video not ready. Status: {job.status}"
        )
    
    if not job.video_url:
        raise HTTPException(status_code=404, detail="Video file not found")
    
    from pathlib import Path
    video_path = Path(job.video_url)
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    
    # Export in background
    platform_exporter = get_platform_exporter()
    background_tasks.add_task(
        platform_exporter.export_for_multiple_platforms,
        video_path,
        platforms
    )
    
    return {
        "video_id": str(video_id),
        "platforms": platforms,
        "status": "exporting",
        "message": "Export started in background",
    }


@router.get("/platforms")
async def list_platforms():
    """List all supported platforms"""
    platform_exporter = get_platform_exporter()
    specs = platform_exporter.get_platform_specs()
    
    return {
        "platforms": [
            {
                "name": name,
                **specs_data
            }
            for name, specs_data in specs.items()
        ]
    }


@router.get("/recommendations")
async def get_recommendations(
    script_text: str,
    language: str = "es",
    platform: Optional[str] = None,
    content_type: str = "general"
):
    """
    Get intelligent recommendations for video generation
    
    Args:
        script_text: Script text
        language: Language code
        platform: Target platform
        content_type: Content type
        
    Returns:
        Recommendations
    """
    recommendation_service = get_recommendation_service()
    recommendations = recommendation_service.get_full_recommendations(
        script_text=script_text,
        language=language,
        platform=platform,
        content_type=content_type
    )
    
    return recommendations


@router.get("/music/tracks")
async def list_music_tracks(style: Optional[str] = None):
    """List available music tracks"""
    music_lib = get_music_library()
    
    if style:
        from ..services.music_library import MusicStyle
        try:
            style_enum = MusicStyle(style)
            tracks = music_lib.get_tracks_by_style(style_enum)
            return {"tracks": [t.to_dict() for t in tracks]}
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid style: {style}")
    
    tracks = music_lib.list_tracks()
    return {"tracks": tracks}


@router.post("/custom-templates")
async def create_custom_template(
    name: str,
    description: str,
    config: Dict[str, Any],
    is_public: bool = False,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Create custom template"""
    from ..services.auth.user_service import get_user_service
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    custom_template_service = get_custom_template_service()
    template = custom_template_service.create_template(
        user_id=user.user_id,
        name=name,
        description=description,
        config=config,
        is_public=is_public
    )
    
    return template.to_dict()


@router.get("/custom-templates")
async def list_custom_templates(
    user_only: bool = False,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """List custom templates"""
    custom_template_service = get_custom_template_service()
    
    if user_only and x_api_key:
        from ..services.auth.user_service import get_user_service
        user_service = get_user_service()
        user = user_service.get_user_by_api_key(x_api_key)
        
        if user:
            templates = custom_template_service.get_user_templates(user.user_id)
            return {"templates": [t.to_dict() for t in templates]}
    
    # Get public templates
    templates = custom_template_service.get_public_templates()
    return {"templates": [t.to_dict() for t in templates]}


@router.post("/ab-tests")
async def create_ab_test(
    name: str,
    variants: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None
):
    """Create A/B test"""
    ab_service = get_ab_testing_service()
    test = ab_service.create_test(name=name, variants=variants, metrics=metrics)
    return test.to_dict()


@router.get("/ab-tests/{test_id}/results")
async def get_ab_test_results(test_id: str):
    """Get A/B test results"""
    ab_service = get_ab_testing_service()
    
    try:
        results = ab_service.get_test_results(test_id)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/videos/{video_id}/share")
async def share_video(
    video_id: UUID,
    shared_with_email: Optional[str] = None,
    shared_with_id: Optional[str] = None,
    permission: str = "view",
    is_public: bool = False,
    expires_at: Optional[datetime] = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Share video with user or make public"""
    from ..services.auth.user_service import get_user_service
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        perm = SharePermission(permission)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid permission: {permission}")
    
    collaboration_service = get_collaboration_service()
    share = collaboration_service.share_video(
        video_id=video_id,
        owner_id=user.user_id,
        shared_with_id=shared_with_id,
        shared_with_email=shared_with_email,
        permission=perm,
        is_public=is_public,
        expires_at=expires_at
    )
    
    return share.to_dict()


@router.get("/videos/{video_id}/shares")
async def get_video_shares(video_id: UUID):
    """Get all shares for a video"""
    collaboration_service = get_collaboration_service()
    shares = collaboration_service.get_video_shares(video_id)
    return {"shares": [s.to_dict() for s in shares]}


@router.get("/shared-videos")
async def get_shared_videos(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Get videos shared with current user"""
    from ..services.auth.user_service import get_user_service
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    collaboration_service = get_collaboration_service()
    shares = collaboration_service.get_user_shared_videos(user.user_id)
    return {"videos": [s.to_dict() for s in shares]}


@router.get("/shared/{share_token}")
async def get_shared_video_by_token(share_token: str):
    """Get shared video by public token"""
    collaboration_service = get_collaboration_service()
    share = collaboration_service.get_share_by_token(share_token)
    
    if not share:
        raise HTTPException(status_code=404, detail="Share not found or expired")
    
    return share.to_dict()


@router.post("/videos/{video_id}/schedule")
async def schedule_video(
    video_id: UUID,
    scheduled_at: datetime,
    request: VideoGenerationRequest,
    timezone: str = "UTC",
    repeat: Optional[str] = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Schedule video generation for future"""
    from ..services.auth.user_service import get_user_service
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    scheduler_service = get_scheduler_service()
    job = scheduler_service.schedule_video(
        video_id=video_id,
        request=request,
        scheduled_at=scheduled_at,
        timezone=timezone,
        repeat=repeat
    )
    
    return job.to_dict()


@router.get("/scheduled")
async def get_scheduled_videos(
    video_id: Optional[UUID] = None,
    status: Optional[str] = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Get scheduled video jobs"""
    scheduler_service = get_scheduler_service()
    jobs = scheduler_service.get_scheduled_jobs(video_id=video_id, status=status)
    return {"jobs": [j.to_dict() for j in jobs]}


@router.delete("/scheduled/{job_id}")
async def cancel_scheduled_job(job_id: str):
    """Cancel scheduled job"""
    scheduler_service = get_scheduler_service()
    success = scheduler_service.cancel_scheduled_job(job_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or already executed")
    
    return {"message": "Job cancelled successfully"}


@router.post("/videos/{video_id}/watermark")
async def add_watermark(
    video_id: UUID,
    watermark_text: Optional[str] = None,
    watermark_image: Optional[str] = None,
    position: str = "bottom-right",
    opacity: float = 0.7,
    size: float = 0.1,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Add watermark to video"""
    if video_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Video job not found")
    
    job = video_jobs[video_id]
    
    if job.status != VideoStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Video not ready. Status: {job.status}"
        )
    
    if not job.video_url:
        raise HTTPException(status_code=404, detail="Video file not found")
    
    from pathlib import Path
    video_path = Path(job.video_url)
    watermark_image_path = Path(watermark_image) if watermark_image else None
    
    watermark_service = get_watermark_service()
    
    # Process in background
    background_tasks.add_task(
        watermark_service.add_watermark,
        video_path,
        watermark_text,
        watermark_image_path,
        position,
        opacity,
        size
    )
    
    return {
        "video_id": str(video_id),
        "status": "processing",
        "message": "Watermarking started in background",
    }


@router.post("/videos/{video_id}/transcribe")
async def transcribe_video(
    video_id: UUID,
    language: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Transcribe video audio to text"""
    if video_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Video job not found")
    
    job = video_jobs[video_id]
    
    if job.status != VideoStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Video not ready. Status: {job.status}"
        )
    
    if not job.video_url:
        raise HTTPException(status_code=404, detail="Video file not found")
    
    from pathlib import Path
    video_path = Path(job.video_url)
    
    transcription_service = get_transcription_service()
    
    # Process in background
    result = await transcription_service.transcribe_video(video_path, language)
    
    return {
        "video_id": str(video_id),
        "transcription": result,
    }


@router.post("/videos/{video_id}/effects/ken-burns")
async def add_ken_burns_effect(
    video_id: UUID,
    zoom: float = 1.2,
    pan_x: float = 0.1,
    pan_y: float = 0.1,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Add Ken Burns effect to video"""
    if video_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Video job not found")
    
    job = video_jobs[video_id]
    
    if not job.video_url:
        raise HTTPException(status_code=404, detail="Video file not found")
    
    from pathlib import Path
    video_path = Path(job.video_url)
    
    visual_effects_service = get_visual_effects_service()
    
    # Process in background
    background_tasks.add_task(
        visual_effects_service.add_ken_burns_effect,
        video_path,
        job.duration or 10.0,
        None,
        zoom,
        pan_x,
        pan_y
    )
    
    return {
        "video_id": str(video_id),
        "status": "processing",
        "message": "Ken Burns effect started in background",
    }


@router.get("/admin/dashboard")
async def get_admin_dashboard(
    time_range_days: int = 30,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Get admin dashboard data"""
    from ..services.auth.user_service import get_user_service
    from ..services.auth.permissions import Permission, check_permission
    from datetime import timedelta
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not check_permission(user.roles, Permission.VIEW_ANALYTICS):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    dashboard_service = get_dashboard_service()
    time_range = timedelta(days=time_range_days)
    data = dashboard_service.get_dashboard_data(time_range=time_range)
    
    return data


@router.get("/admin/health")
async def get_system_health():
    """Get system health status"""
    monitoring_service = get_monitoring_service()
    health_checks = monitoring_service.run_health_checks()
    return health_checks


@router.get("/admin/metrics")
async def get_system_metrics():
    """Get system metrics"""
    monitoring_service = get_monitoring_service()
    metrics = monitoring_service.get_system_metrics()
    return metrics


@router.get("/admin/profiles")
async def get_performance_profiles():
    """Get performance profiles"""
    profiler_service = get_profiler_service()
    profiles = profiler_service.get_all_profiles()
    return {"profiles": profiles}


@router.post("/admin/backup")
async def create_backup(
    include_videos: bool = True,
    include_metadata: bool = True,
    include_config: bool = True,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Create system backup"""
    from ..services.auth.user_service import get_user_service
    from ..services.auth.permissions import Permission, check_permission
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not check_permission(user.roles, Permission.MANAGE_SETTINGS):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    backup_service = get_backup_service()
    backup_info = await backup_service.create_backup(
        include_videos=include_videos,
        include_metadata=include_metadata,
        include_config=include_config
    )
    
    return backup_info


@router.get("/admin/backups")
async def list_backups(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """List all backups"""
    from ..services.auth.user_service import get_user_service
    from ..services.auth.permissions import Permission, check_permission
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not check_permission(user.roles, Permission.MANAGE_SETTINGS):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    backup_service = get_backup_service()
    backups = backup_service.list_backups()
    return {"backups": backups}


@router.post("/admin/backups/{backup_id}/restore")
async def restore_backup(
    backup_id: str,
    restore_videos: bool = True,
    restore_metadata: bool = True,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Restore from backup"""
    from ..services.auth.user_service import get_user_service
    from ..services.auth.permissions import Permission, check_permission
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not check_permission(user.roles, Permission.MANAGE_SETTINGS):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    backup_service = get_backup_service()
    success = await backup_service.restore_backup(
        backup_id=backup_id,
        restore_videos=restore_videos,
        restore_metadata=restore_metadata
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Backup not found or restore failed")
    
    return {"message": "Backup restored successfully"}


@router.get("/admin/config")
async def get_configuration(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Get system configuration"""
    from ..services.auth.user_service import get_user_service
    from ..services.auth.permissions import Permission, check_permission
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not check_permission(user.roles, Permission.MANAGE_SETTINGS):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    config_manager = get_config_manager()
    return config_manager.get_all()


@router.put("/admin/config")
async def update_configuration(
    config_updates: Dict[str, Any],
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Update system configuration"""
    from ..services.auth.user_service import get_user_service
    from ..services.auth.permissions import Permission, check_permission
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not check_permission(user.roles, Permission.MANAGE_SETTINGS):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    config_manager = get_config_manager()
    config_manager.update(config_updates)
    
    return {"message": "Configuration updated", "config": config_manager.get_all()}


@router.get("/quota")
async def get_user_quota(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Get user quota information"""
    from ..services.auth.user_service import get_user_service
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    advanced_limiter = get_advanced_rate_limiter()
    quota_info = advanced_limiter.get_quota_info(user.user_id)
    
    return quota_info


@router.post("/videos/{video_id}/feedback")
async def submit_feedback(
    video_id: UUID,
    rating: int,
    comment: Optional[str] = None,
    tags: Optional[List[str]] = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Submit feedback for video"""
    from ..services.auth.user_service import get_user_service
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    feedback_service = get_feedback_service()
    
    try:
        feedback = feedback_service.submit_feedback(
            video_id=video_id,
            user_id=user.user_id,
            rating=rating,
            comment=comment,
            tags=tags
        )
        return feedback.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/videos/{video_id}/feedback")
async def get_video_feedback(video_id: UUID):
    """Get feedback for video"""
    feedback_service = get_feedback_service()
    feedbacks = feedback_service.get_video_feedback(video_id)
    stats = feedback_service.get_feedback_stats(video_id)
    
    return {
        "feedbacks": [f.to_dict() for f in feedbacks],
        "statistics": stats,
    }


@router.get("/search")
async def search_videos(
    q: str,
    status: Optional[str] = None,
    tags: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50
):
    """Search videos"""
    search_service = get_search_service()
    
    filters = {}
    if status:
        filters["status"] = status
    if tags:
        filters["tags"] = tags.split(",")
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to
    
    results = search_service.search_videos(query=q, filters=filters, limit=limit)
    return {"results": results, "total": len(results)}


@router.get("/search/suggestions")
async def get_search_suggestions(q: str, limit: int = 5):
    """Get search suggestions"""
    search_service = get_search_service()
    suggestions = search_service.get_suggestions(q, limit=limit)
    return {"suggestions": suggestions}


@router.get("/export/videos")
async def export_videos_data(
    format: str = "csv",
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Export videos data"""
    from ..services.auth.user_service import get_user_service
    from ..services.auth.permissions import Permission, check_permission
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not check_permission(user.roles, Permission.VIEW_ANALYTICS):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Get all videos (in production, from database)
    videos_data = []
    for video_id, job in video_jobs.items():
        videos_data.append({
            "video_id": str(video_id),
            "status": job.status.value if hasattr(job.status, 'value') else str(job.status),
            "created_at": job.created_at.isoformat() if hasattr(job, 'created_at') else None,
            "duration": job.duration,
            "file_size": job.file_size,
        })
    
    data_exporter = get_data_exporter()
    
    if format.lower() == "csv":
        export_path = data_exporter.export_to_csv(videos_data)
        return FileResponse(
            export_path,
            media_type="text/csv",
            filename=export_path.name
        )
    elif format.lower() == "json":
        export_path = data_exporter.export_to_json(videos_data)
        return FileResponse(
            export_path,
            media_type="application/json",
            filename=export_path.name
        )
    else:
        raise HTTPException(status_code=400, detail="Format must be 'csv' or 'json'")


@router.get("/metrics/prometheus")
async def get_prometheus_metrics_endpoint():
    """Get Prometheus metrics"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    metrics = get_prometheus_metrics()
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@router.get("/alerts")
async def get_alerts(
    severity: Optional[str] = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Get active alerts"""
    from ..services.auth.user_service import get_user_service
    from ..services.auth.permissions import Permission, check_permission
    from ..services.alerts import AlertSeverity
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not check_permission(user.roles, Permission.VIEW_ANALYTICS):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    alert_manager = get_alert_manager()
    
    severity_enum = None
    if severity:
        try:
            severity_enum = AlertSeverity(severity)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
    
    alerts = alert_manager.get_active_alerts(severity=severity_enum)
    return {"alerts": [a.to_dict() for a in alerts]}


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Acknowledge alert"""
    from ..services.auth.user_service import get_user_service
    from ..services.auth.permissions import Permission, check_permission
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not check_permission(user.roles, Permission.MANAGE_SETTINGS):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    alert_manager = get_alert_manager()
    alert_manager.acknowledge_alert(alert_id)
    
    return {"message": "Alert acknowledged"}


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Resolve alert"""
    from ..services.auth.user_service import get_user_service
    from ..services.auth.permissions import Permission, check_permission
    
    user_service = get_user_service()
    user = user_service.get_user_by_api_key(x_api_key) if x_api_key else None
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not check_permission(user.roles, Permission.MANAGE_SETTINGS):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    alert_manager = get_alert_manager()
    alert_manager.resolve_alert(alert_id)
    
    return {"message": "Alert resolved"}

