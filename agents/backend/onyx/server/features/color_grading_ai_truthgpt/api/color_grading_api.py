"""
REST API for Color Grading AI TruthGPT
=======================================

FastAPI REST API for color grading operations.
"""

import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import uuid
import asyncio
import time

from ..core.color_grading_agent import ColorGradingAgent
from ..config.color_grading_config import ColorGradingConfig
from .middleware import (
    rate_limit_middleware,
    request_logging_middleware,
    error_handling_middleware
)
from .health_check import HealthChecker
from .openapi_extensions import custom_openapi
from .dashboard import router as dashboard_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Color Grading AI TruthGPT",
    description="Automatic color grading API with OpenRouter and TruthGPT",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.middleware("http")(error_handling_middleware)
app.middleware("http")(request_logging_middleware)
app.middleware("http")(rate_limit_middleware)

# Custom OpenAPI schema
app.openapi = lambda: custom_openapi(app)

# Include dashboard router
app.include_router(dashboard_router)

# Global agent instance
agent: Optional[ColorGradingAgent] = None
health_checker: Optional[HealthChecker] = None


@app.on_event("startup")
async def startup():
    """Initialize agent on startup."""
    global agent, health_checker
    config = ColorGradingConfig()
    config.validate()
    agent = ColorGradingAgent(config=config)
    health_checker = HealthChecker(agent=agent)
    logger.info("Color Grading API started")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global agent
    if agent:
        await agent.close()
    logger.info("Color Grading API stopped")


# Request/Response Models
class ColorGradingRequest(BaseModel):
    """Request model for color grading."""
    template_name: Optional[str] = None
    description: Optional[str] = None
    color_params: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseModel):
    """Response model for analysis."""
    success: bool
    analysis: Dict[str, Any]


class GradingResponse(BaseModel):
    """Response model for color grading."""
    success: bool
    output_path: str
    color_params: Dict[str, Any]
    task_id: Optional[str] = None


# API Endpoints
@app.post("/api/v1/grade/video", response_model=GradingResponse)
async def grade_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    template_name: Optional[str] = None,
    reference_image: Optional[UploadFile] = File(None),
    reference_video: Optional[UploadFile] = File(None),
    description: Optional[str] = None,
    color_params: Optional[str] = None
):
    """
    Apply color grading to video.
    
    Args:
        file: Input video file
        template_name: Template name to apply
        reference_image: Reference image for color matching
        reference_video: Reference video for color matching
        description: Text description of desired look
        color_params: JSON string with color parameters
        
    Returns:
        Grading response with output path
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Save uploaded file
    task_id = str(uuid.uuid4())
    input_dir = agent.output_dirs["storage"] / "uploads"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = input_dir / f"{task_id}_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Save reference files if provided
    ref_image_path = None
    ref_video_path = None
    
    if reference_image:
        ref_image_path = input_dir / f"{task_id}_ref_image_{reference_image.filename}"
        with open(ref_image_path, "wb") as f:
            content = await reference_image.read()
            f.write(content)
    
    if reference_video:
        ref_video_path = input_dir / f"{task_id}_ref_video_{reference_video.filename}"
        with open(ref_video_path, "wb") as f:
            content = await reference_video.read()
            f.write(content)
    
    # Parse color_params if provided
    params = None
    if color_params:
        import json
        params = json.loads(color_params)
    
    try:
        result = await agent.grade_video(
            video_path=str(input_path),
            template_name=template_name,
            reference_image=str(ref_image_path) if ref_image_path else None,
            reference_video=str(ref_video_path) if ref_video_path else None,
            color_params=params,
            description=description
        )
        
        return GradingResponse(
            success=True,
            output_path=result["output_path"],
            color_params=result["color_params"],
            task_id=task_id
        )
    except Exception as e:
        logger.error(f"Error grading video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/grade/image", response_model=GradingResponse)
async def grade_image(
    file: UploadFile = File(...),
    template_name: Optional[str] = None,
    reference_image: Optional[UploadFile] = File(None),
    description: Optional[str] = None,
    color_params: Optional[str] = None
):
    """
    Apply color grading to image.
    
    Args:
        file: Input image file
        template_name: Template name to apply
        reference_image: Reference image for color matching
        description: Text description of desired look
        color_params: JSON string with color parameters
        
    Returns:
        Grading response with output path
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Save uploaded file
    task_id = str(uuid.uuid4())
    input_dir = agent.output_dirs["storage"] / "uploads"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = input_dir / f"{task_id}_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Save reference image if provided
    ref_image_path = None
    if reference_image:
        ref_image_path = input_dir / f"{task_id}_ref_{reference_image.filename}"
        with open(ref_image_path, "wb") as f:
            content = await reference_image.read()
            f.write(content)
    
    # Parse color_params if provided
    params = None
    if color_params:
        import json
        params = json.loads(color_params)
    
    try:
        result = await agent.grade_image(
            image_path=str(input_path),
            template_name=template_name,
            reference_image=str(ref_image_path) if ref_image_path else None,
            color_params=params,
            description=description
        )
        
        return GradingResponse(
            success=True,
            output_path=result["output_path"],
            color_params=result["color_params"],
            task_id=task_id
        )
    except Exception as e:
        logger.error(f"Error grading image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_media(
    file: UploadFile = File(...),
    media_type: str = "auto"
):
    """
    Analyze color properties of media.
    
    Args:
        file: Input media file
        media_type: "image", "video", or "auto"
        
    Returns:
        Analysis results
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Save uploaded file
    task_id = str(uuid.uuid4())
    input_dir = agent.output_dirs["storage"] / "uploads"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = input_dir / f"{task_id}_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        analysis = await agent.analyze_media(str(input_path), media_type)
        
        return AnalysisResponse(
            success=True,
            analysis=analysis
        )
    except Exception as e:
        logger.error(f"Error analyzing media: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/templates")
async def list_templates(
    category: Optional[str] = None,
    tags: Optional[str] = None
):
    """
    List available templates.
    
    Args:
        category: Filter by category
        tags: Comma-separated tags to filter
        
    Returns:
        List of templates
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    tag_list = tags.split(",") if tags else None
    
    templates = await agent.list_templates(category=category, tags=tag_list)
    
    return {"templates": templates}


@app.get("/api/v1/download/{task_id}")
async def download_result(task_id: str):
    """
    Download processed result.
    
    Args:
        task_id: Task ID
        
    Returns:
        File response
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Find result file
    result_dir = agent.output_dirs["results"]
    result_files = list(result_dir.glob(f"*{task_id}*"))
    
    if not result_files:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return FileResponse(
        path=str(result_files[0]),
        filename=result_files[0].name,
        media_type="application/octet-stream"
    )


@app.post("/api/v1/batch/process")
async def process_batch(
    items: List[Dict[str, Any]],
    media_type: str = "video"
):
    """
    Process batch of media files.
    
    Args:
        items: List of items with input_path, output_path, parameters
        media_type: "video" or "image"
        
    Returns:
        Batch job ID
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    job_id = await agent.process_batch(items, media_type)
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/v1/batch/{job_id}")
async def get_batch_status(job_id: str):
    """Get batch job status."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    status = await agent.get_batch_status(job_id)
    return status


@app.post("/api/v1/webhooks/register")
async def register_webhook(
    url: str,
    events: List[str],
    secret: Optional[str] = None
):
    """Register webhook for notifications."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    agent.register_webhook(url, events, secret)
    return {"status": "registered", "url": url}


@app.post("/api/v1/comparison")
async def create_comparison(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
    style: str = "side_by_side"
):
    """Create before/after comparison."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Save files
    task_id = str(uuid.uuid4())
    input_dir = agent.output_dirs["storage"] / "uploads"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    before_path = input_dir / f"{task_id}_before_{before_file.filename}"
    after_path = input_dir / f"{task_id}_after_{after_file.filename}"
    
    with open(before_path, "wb") as f:
        f.write(await before_file.read())
    with open(after_path, "wb") as f:
        f.write(await after_file.read())
    
    output_path = str(agent.output_dirs["results"] / f"comparison_{task_id}.jpg")
    
    result_path = await agent.create_comparison(
        str(before_path),
        str(after_path),
        output_path,
        style
    )
    
    return {"success": True, "output_path": result_path}


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get processing metrics."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    metrics = agent.get_metrics()
    return metrics


@app.get("/api/v1/luts")
async def list_luts(
    category: Optional[str] = None,
    format: Optional[str] = None
):
    """List available LUTs."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    luts = agent.lut_manager.list_luts(category=category, format=format)
    return {"luts": [lut.to_dict() for lut in luts]}


@app.post("/api/v1/tasks/enqueue")
async def enqueue_task(
    task_type: str,
    parameters: Dict[str, Any],
    priority: str = "NORMAL"
):
    """Enqueue task for async processing."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    from ..services.task_queue import TaskPriority
    try:
        priority_enum = TaskPriority[priority.upper()]
    except KeyError:
        priority_enum = TaskPriority.NORMAL
    
    task_id = await agent.enqueue_task(task_type, parameters, priority_enum)
    return {"task_id": task_id, "status": "queued"}


@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    status = await agent.get_task_status(task_id)
    return status


@app.get("/api/v1/queue/status")
async def get_queue_status():
    """Get queue status."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    status = await agent.task_queue.get_queue_status()
    return status


@app.post("/api/v1/export/parameters")
async def export_parameters(
    parameters: Dict[str, Any],
    format: str = "all"
):
    """Export color parameters."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    task_id = str(uuid.uuid4())
    output_path = str(agent.output_dirs["results"] / f"params_{task_id}")
    
    exports = agent.export_parameters(parameters, output_path, format)
    return {"exports": exports}


@app.get("/api/v1/history")
async def get_history(
    operation: Optional[str] = None,
    template: Optional[str] = None,
    limit: int = 100
):
    """Get processing history."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    history = agent.get_history(operation=operation, template=template, limit=limit)
    return {"history": history}


@app.get("/api/v1/resources")
async def get_resources():
    """Get system resource statistics."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    stats = agent.get_resource_stats()
    return stats


@app.get("/health")
async def health():
    """Basic health check endpoint."""
    return {"status": "healthy", "agent_initialized": agent is not None}


@app.get("/health/detailed")
async def health_detailed():
    """Detailed health check endpoint."""
    if not health_checker:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "Health checker not initialized"}
        )
    
    health_status = health_checker.get_full_health()
    health_status["timestamp"] = datetime.utcnow().isoformat()
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(status_code=status_code, content=health_status)


@app.get("/health/system")
async def health_system():
    """System health check."""
    if not health_checker:
        return JSONResponse(status_code=503, content={"error": "Health checker not initialized"})
    
    return health_checker.check_system()


@app.get("/health/agent")
async def health_agent():
    """Agent health check."""
    if not health_checker:
        return JSONResponse(status_code=503, content={"error": "Health checker not initialized"})
    
    return health_checker.check_agent()


@app.get("/health/dependencies")
async def health_dependencies():
    """Dependencies health check."""
    if not health_checker:
        return JSONResponse(status_code=503, content={"error": "Health checker not initialized"})
    
    return health_checker.check_dependencies()


@app.post("/api/v1/analyze/quality")
async def analyze_video_quality(file: UploadFile = File(...)):
    """Analyze video quality metrics."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Save uploaded file
    task_id = str(uuid.uuid4())
    input_dir = agent.output_dirs["storage"] / "uploads"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = input_dir / f"{task_id}_{file.filename}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        analysis = await agent.analyze_video_quality(str(input_path))
        return {"success": True, "analysis": analysis}
    except Exception as e:
        logger.error(f"Error analyzing video quality: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/presets")
async def create_preset(
    name: str,
    description: str,
    color_params: Dict[str, Any],
    category: Optional[str] = None,
    tags: Optional[List[str]] = None
):
    """Create a color grading preset."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    preset_id = agent.create_preset(name, description, color_params, category, tags)
    return {"preset_id": preset_id, "status": "created"}


@app.get("/api/v1/presets")
async def list_presets(
    category: Optional[str] = None,
    tags: Optional[str] = None,
    favorites_only: bool = False
):
    """List color grading presets."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    tag_list = tags.split(",") if tags else None
    presets = agent.list_presets(category=category, tags=tag_list, favorites_only=favorites_only)
    return {"presets": presets}


@app.post("/api/v1/backup/create")
async def create_backup(source_dirs: List[str], backup_name: Optional[str] = None):
    """Create a backup."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    backup_path = agent.create_backup(source_dirs, backup_name)
    return {"backup_path": backup_path, "status": "created"}


@app.get("/api/v1/backup/list")
async def list_backups():
    """List available backups."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    backups = agent.list_backups()
    return {"backups": backups}


@app.get("/api/v1/plugins")
async def list_plugins():
    """List available plugins."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    plugins = agent.list_plugins()
    return {"plugins": plugins}


@app.post("/api/v1/auth/keys")
async def create_api_key(
    name: str,
    permissions: List[str],
    expires_days: Optional[int] = None
):
    """Create a new API key."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    api_key = agent.auth_manager.generate_api_key(name, permissions, expires_days)
    return {"api_key": api_key, "name": name, "permissions": permissions}


@app.get("/api/v1/auth/keys")
async def list_api_keys():
    """List API keys."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    keys = agent.auth_manager.list_api_keys()
    return {"keys": keys}


@app.post("/api/v1/auth/keys/{key_id}/revoke")
async def revoke_api_key(key_id: str):
    """Revoke an API key."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    success = agent.auth_manager.revoke_api_key(key_id)
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")
    
    return {"status": "revoked", "key_id": key_id}


@app.post("/api/v1/versions/create")
async def create_version(
    media_id: str,
    color_params: Dict[str, Any],
    parent_id: Optional[str] = None,
    description: Optional[str] = None
):
    """Create a new version of color parameters."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    version_id = agent.version_manager.create_version(
        media_id, color_params, parent_id, description
    )
    return {"version_id": version_id, "status": "created"}


@app.get("/api/v1/versions/{media_id}")
async def get_versions(media_id: str):
    """Get all versions for media."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    versions = agent.version_manager.get_versions(media_id)
    return {"versions": [v.to_dict() for v in versions]}


@app.post("/api/v1/versions/{media_id}/rollback/{version_id}")
async def rollback_version(media_id: str, version_id: str):
    """Rollback to a specific version."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    success = agent.version_manager.rollback_to_version(media_id, version_id)
    if not success:
        raise HTTPException(status_code=404, detail="Version not found")
    
    return {"status": "rolled_back", "version_id": version_id}


@app.get("/api/v1/versions/{media_id}/compare")
async def compare_versions(
    media_id: str,
    version_id1: str,
    version_id2: str
):
    """Compare two versions."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    comparison = agent.version_manager.compare_versions(media_id, version_id1, version_id2)
    return comparison


@app.post("/api/v1/cloud/upload")
async def upload_to_cloud(
    provider: str,
    local_path: str,
    remote_path: str
):
    """Upload file to cloud storage."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    cloud_url = await agent.cloud_integration.upload_to_cloud(
        provider, local_path, remote_path
    )
    return {"cloud_url": cloud_url, "status": "uploaded"}


@app.get("/api/v1/cloud/providers")
async def list_cloud_providers():
    """List registered cloud providers."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    providers = agent.cloud_integration.list_providers()
    return {"providers": providers}

