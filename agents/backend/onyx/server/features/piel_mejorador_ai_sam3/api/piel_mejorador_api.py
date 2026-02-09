"""
REST API for Piel Mejorador AI SAM3
====================================

REST API wrapper for the skin enhancement agent.
"""

import asyncio
import logging
import os
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..core.piel_mejorador_agent import PielMejoradorAgent
from ..config.piel_mejorador_config import PielMejoradorConfig
from ..core.rate_limiter import RateLimiter, RateLimitConfig
from ..core.webhook_manager import Webhook, WebhookEvent
from .api_helpers import require_agent, handle_task_operation
from .error_handlers import handle_task_errors
from .response_builder import ResponseBuilder
from fastapi import Request, HTTPException, status, Depends, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Piel Mejorador AI SAM3 API",
    version="1.0.0",
    root_path="/api/v1",
    description="""
    ## Piel Mejorador AI SAM3 API
    
    Sistema avanzado de mejoramiento de piel con arquitectura SAM3.
    
    ### Características:
    - ✅ Procesamiento de imágenes y videos
    - ✅ Niveles configurables de mejora y realismo
    - ✅ Procesamiento frame-by-frame para videos
    - ✅ Sistema de caché inteligente
    - ✅ Procesamiento en lote
    - ✅ Rate limiting
    - ✅ Webhooks para notificaciones
    - ✅ Optimización automática de memoria
    
    ### Autenticación:
    Configure `OPENROUTER_API_KEY` como variable de entorno.
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Enhancement",
            "description": "Operaciones de mejoramiento de piel",
        },
        {
            "name": "Analysis",
            "description": "Análisis de condición de piel",
        },
        {
            "name": "Batch",
            "description": "Procesamiento en lote",
        },
        {
            "name": "Webhooks",
            "description": "Gestión de webhooks",
        },
        {
            "name": "Monitoring",
            "description": "Métricas y monitoreo",
        },
    ]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
_agent: Optional[PielMejoradorAgent] = None

# Rate limiter
_rate_limiter = RateLimiter(
    default_config=RateLimitConfig(
        requests_per_second=10.0,
        burst_size=20
    )
)

# Authentication (optional, can be disabled)
_auth_enabled = os.getenv("PIEL_MEJORADOR_AUTH_ENABLED", "false").lower() == "true"
_auth_manager = None

if _auth_enabled:
    from ..core.auth_manager import AuthManager
    _auth_manager = AuthManager(secret_key=os.getenv("PIEL_MEJORADOR_SECRET_KEY"))

security = HTTPBearer(auto_error=False)


async def verify_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[Dict]:
    """
    Verify authentication.
    
    Supports both API key (X-API-Key header) and JWT (Bearer token).
    """
    if not _auth_enabled or not _auth_manager:
        return None  # Auth disabled
    
    # Try API key first
    if x_api_key:
        api_key = _auth_manager.validate_api_key(x_api_key)
        if api_key:
            return {
                "key_id": api_key.key_id,
                "permissions": api_key.permissions
            }
    
    # Try JWT token
    if credentials:
        payload = _auth_manager.validate_jwt(credentials.credentials)
        if payload:
            return payload
    
    # No valid auth found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    # Get client identifier (IP address)
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    if not await _rate_limiter.is_allowed(client_ip):
        wait_time = await _rate_limiter.get_wait_time(client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please wait {wait_time:.2f} seconds.",
            headers={"Retry-After": str(int(wait_time))}
        )
    
    response = await call_next(request)
    return response


# Request models
class MejorarImagenRequest(BaseModel):
    file_path: str
    enhancement_level: str = "medium"
    realism_level: Optional[float] = None
    custom_instructions: Optional[str] = None
    priority: int = 0


class MejorarVideoRequest(BaseModel):
    file_path: str
    enhancement_level: str = "medium"
    realism_level: Optional[float] = None
    custom_instructions: Optional[str] = None
    priority: int = 0


class AnalizarPielRequest(BaseModel):
    file_path: str
    file_type: str = "image"
    priority: int = 0


class BatchItemRequest(BaseModel):
    file_path: str
    enhancement_level: str = "medium"
    realism_level: Optional[float] = None
    custom_instructions: Optional[str] = None
    priority: int = 0
    metadata: Optional[Dict[str, Any]] = None


class BatchProcessRequest(BaseModel):
    items: List[BatchItemRequest]


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup."""
    global _agent
    config = PielMejoradorConfig()
    _agent = PielMejoradorAgent(config=config)
    logger.info("PielMejoradorAgent initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _agent
    if _agent:
        await _agent.close()
        _agent = None


@app.post("/upload-image", tags=["Enhancement"])
async def upload_image(
    auth: Optional[Dict] = Depends(verify_auth),
    file: UploadFile = File(...),
    enhancement_level: str = Form("medium"),
    realism_level: Optional[float] = Form(None),
    custom_instructions: Optional[str] = Form(None),
    priority: int = Form(0)
):
    """Upload and enhance an image."""
    agent = require_agent(_agent)
    
    # Save uploaded file
    upload_dir = agent.output_dirs["uploads"]
    file_path = upload_dir / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate file
        from ..core.helpers import validate_image_file
        validate_image_file(str(file_path), agent.config.max_image_size_mb)
        
        # Submit task
        task_id = await agent.mejorar_imagen(
            file_path=str(file_path),
            enhancement_level=enhancement_level,
            realism_level=realism_level,
            custom_instructions=custom_instructions,
            priority=priority
        )
        
        return ResponseBuilder.task_submitted(task_id)
    except Exception as e:
        logger.error(f"Error uploading image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    enhancement_level: str = Form("medium"),
    realism_level: Optional[float] = Form(None),
    custom_instructions: Optional[str] = Form(None),
    priority: int = Form(0)
):
    """Upload and enhance a video."""
    agent = require_agent(_agent)
    
    # Save uploaded file
    upload_dir = agent.output_dirs["uploads"]
    file_path = upload_dir / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate file
        from ..core.helpers import validate_video_file
        validate_video_file(str(file_path), agent.config.max_video_size_mb)
        
        # Submit task
        task_id = await agent.mejorar_video(
            file_path=str(file_path),
            enhancement_level=enhancement_level,
            realism_level=realism_level,
            custom_instructions=custom_instructions,
            priority=priority
        )
        
        return ResponseBuilder.task_submitted(task_id)
    except Exception as e:
        logger.error(f"Error uploading video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mejorar-imagen", tags=["Enhancement"])
async def mejorar_imagen(
    request: MejorarImagenRequest,
    auth: Optional[Dict] = Depends(verify_auth)
):
    """Enhance an image from file path."""
    agent = require_agent(_agent)
    
    task_id = await agent.mejorar_imagen(
        file_path=request.file_path,
        enhancement_level=request.enhancement_level,
        realism_level=request.realism_level,
        custom_instructions=request.custom_instructions,
        priority=request.priority
    )
    
    return ResponseBuilder.task_submitted(task_id)


@app.post("/mejorar-video")
async def mejorar_video(request: MejorarVideoRequest):
    """Enhance a video from file path."""
    agent = require_agent(_agent)
    
    task_id = await agent.mejorar_video(
        file_path=request.file_path,
        enhancement_level=request.enhancement_level,
        realism_level=request.realism_level,
        custom_instructions=request.custom_instructions,
        priority=request.priority
    )
    
    return ResponseBuilder.task_submitted(task_id)


@app.post("/analizar-piel")
async def analizar_piel(request: AnalizarPielRequest):
    """Analyze skin condition."""
    agent = require_agent(_agent)
    
    task_id = await agent.analizar_piel(
        file_path=request.file_path,
        file_type=request.file_type,
        priority=request.priority
    )
    
    return ResponseBuilder.task_submitted(task_id)


@app.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    """Get task status."""
    agent = require_agent(_agent)
    
    return await handle_task_operation(
        agent,
        "get_task_status",
        agent.get_task_status,
        task_id
    )


@app.get("/task/{task_id}/result")
@handle_task_errors
async def get_task_result(task_id: str):
    """
    Get task result.
    
    Uses helper for consistent error handling.
    """
    agent = require_agent(_agent)
    result = await agent.get_task_result(task_id)
    
    if result is None:
        raise HTTPException(status_code=404, detail="Task not completed yet")
    
    return result


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Uses ResponseBuilder for consistent response format.
    """
    agent_running = _agent is not None and _agent.running if _agent else False
    return ResponseBuilder.health_check(agent_running)


@app.get("/enhancement-levels")
async def get_enhancement_levels():
    """Get available enhancement levels."""
    agent = require_agent(_agent)
    return {
        "levels": agent.config.enhancement_levels,
        "description": {
            "low": "Mejoras sutiles y naturales",
            "medium": "Mejoras moderadas manteniendo realismo",
            "high": "Mejoras significativas con alto realismo",
            "ultra": "Mejoras máximas con realismo fotográfico perfecto"
        }
    }


@app.get("/stats")
async def get_stats():
    """Get performance statistics."""
    agent = require_agent(_agent)
    return agent.get_performance_stats()


@app.post("/batch-process")
async def batch_process(request: BatchProcessRequest):
    """Process multiple files in batch."""
    agent = require_agent(_agent)
    
    from ..core.batch_processor import BatchItem
    
    # Convert request items to BatchItem
    batch_items = [
        BatchItem(
            file_path=item.file_path,
            enhancement_level=item.enhancement_level,
            realism_level=item.realism_level,
            custom_instructions=item.custom_instructions,
            priority=item.priority,
            metadata=item.metadata or {}
        )
        for item in request.items
    ]
    
    # Process batch
    result = await agent.process_batch(batch_items)
    
    return {
        "total_items": result.total_items,
        "completed": result.completed,
        "failed": result.failed,
        "success_rate": result.success_rate,
        "duration": result.duration,
        "results": result.results,
        "errors": result.errors
    }


@app.post("/cache/cleanup")
async def cleanup_cache():
    """Clean up expired cache entries."""
    agent = require_agent(_agent)
    cleaned = await agent.cleanup_cache()
    return {"cleaned_entries": cleaned, "message": f"Cleaned up {cleaned} expired cache entries"}


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    agent = require_agent(_agent)
    return agent.cache_manager.get_stats()


@app.post("/webhooks/register")
async def register_webhook(
    url: str,
    events: List[str],
    secret: Optional[str] = None
):
    """Register a webhook for notifications."""
    agent = require_agent(_agent)
    
    # Convert string events to WebhookEvent enum
    webhook_events = []
    for event_str in events:
        try:
            webhook_events.append(WebhookEvent(event_str))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event: {event_str}. Valid events: {[e.value for e in WebhookEvent]}"
            )
    
    agent.register_webhook(url=url, events=webhook_events, secret=secret)
    
    return {"message": "Webhook registered successfully", "url": url, "events": events}


@app.delete("/webhooks/unregister")
async def unregister_webhook(url: str):
    """Unregister a webhook."""
    agent = require_agent(_agent)
    agent.webhook_manager.unregister(url)
    return {"message": "Webhook unregistered successfully"}


@app.get("/webhooks/stats")
async def get_webhook_stats():
    """Get webhook statistics."""
    agent = require_agent(_agent)
    return agent.webhook_manager.get_stats()


@app.get("/memory/usage")
async def get_memory_usage():
    """Get memory usage statistics."""
    agent = require_agent(_agent)
    return agent.memory_optimizer.get_memory_usage()


@app.get("/memory/recommendations")
async def get_memory_recommendations():
    """Get memory optimization recommendations."""
    agent = require_agent(_agent)
    return {"recommendations": agent.get_memory_recommendations()}


@app.post("/memory/optimize")
async def optimize_memory(force: bool = False):
    """Optimize memory usage."""
    agent = require_agent(_agent)
    await agent.optimize_memory(force=force)
    return {"message": "Memory optimization completed"}


@app.get("/rate-limit/stats", tags=["Monitoring"])
async def get_rate_limit_stats():
    """Get rate limiting statistics."""
    return _rate_limiter.get_stats()


@app.post("/backup/create", tags=["Management"])
async def create_backup():
    """Create a backup of tasks and data."""
    agent = require_agent(_agent)
    
    backup_info = await agent.backup_manager.create_backup(
        source_dir=agent.output_dirs["storage"]
    )
    
    return {
        "backup_id": backup_info.backup_id,
        "timestamp": backup_info.timestamp.isoformat(),
        "size_mb": backup_info.size_bytes / (1024 * 1024),
        "task_count": backup_info.task_count,
        "success": backup_info.success,
        "error": backup_info.error
    }


@app.get("/backup/list", tags=["Management"])
async def list_backups():
    """List all available backups."""
    agent = require_agent(_agent)
    backups = agent.backup_manager.list_backups()
    
    return {
        "backups": [
            {
                "backup_id": b.backup_id,
                "timestamp": b.timestamp.isoformat(),
                "size_mb": b.size_bytes / (1024 * 1024),
                "task_count": b.task_count
            }
            for b in backups
        ]
    }


@app.post("/backup/restore/{backup_id}", tags=["Management"])
async def restore_backup(backup_id: str):
    """Restore from a backup."""
    agent = require_agent(_agent)
    
    success = await agent.backup_manager.restore_backup(
        backup_id=backup_id,
        target_dir=agent.output_dirs["storage"]
    )
    
    if success:
        return {"message": f"Backup {backup_id} restored successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Backup {backup_id} not found or restore failed")


@app.get("/backup/stats", tags=["Management"])
async def get_backup_stats():
    """Get backup statistics."""
    agent = require_agent(_agent)
    return agent.backup_manager.get_backup_stats()


@app.post("/backup/cleanup", tags=["Management"])
async def cleanup_backups():
    """Clean up old backups."""
    agent = require_agent(_agent)
    cleaned = await agent.backup_manager.cleanup_old_backups()
    return {"cleaned_backups": cleaned, "message": f"Cleaned up {cleaned} old backups"}


@app.get("/performance/stats", tags=["Monitoring"])
async def get_performance_stats():
    """Get performance optimization statistics."""
    agent = require_agent(_agent)
    return {
        "optimizer_stats": agent.performance_optimizer.get_stats(),
        "metrics": agent.performance_optimizer.get_metrics().__dict__,
        "recommendations": agent.performance_optimizer.get_recommendations()
    }


@app.get("/circuit-breaker/stats", tags=["Monitoring"])
async def get_circuit_breaker_stats():
    """Get circuit breaker statistics."""
    agent = require_agent(_agent)
    return {
        "openrouter": agent.openrouter_circuit.get_stats()
    }


@app.post("/auth/api-keys", tags=["Management"])
async def create_api_key(
    name: str,
    permissions: List[str],
    expires_days: Optional[int] = None
):
    """Create a new API key."""
    if not _auth_enabled or not _auth_manager:
        raise HTTPException(status_code=503, detail="Authentication not enabled")
    
    key_id, plain_key = _auth_manager.create_api_key(
        name=name,
        permissions=permissions,
        expires_days=expires_days
    )
    
    return {
        "key_id": key_id,
        "api_key": plain_key,  # Only shown once!
        "message": "Save this API key securely. It won't be shown again."
    }


@app.get("/auth/api-keys", tags=["Management"])
async def list_api_keys():
    """List all API keys."""
    if not _auth_enabled or not _auth_manager:
        raise HTTPException(status_code=503, detail="Authentication not enabled")
    
    return {"api_keys": _auth_manager.list_api_keys()}


@app.delete("/auth/api-keys/{key_id}", tags=["Management"])
async def revoke_api_key(key_id: str):
    """Revoke an API key."""
    if not _auth_enabled or not _auth_manager:
        raise HTTPException(status_code=503, detail="Authentication not enabled")
    
    if _auth_manager.revoke_api_key(key_id):
        return {"message": f"API key {key_id} revoked"}
    else:
        raise HTTPException(status_code=404, detail="API key not found")


@app.post("/auth/jwt", tags=["Management"])
async def generate_jwt(key_id: str, expires_minutes: int = 60):
    """Generate JWT token from API key."""
    if not _auth_enabled or not _auth_manager:
        raise HTTPException(status_code=503, detail="Authentication not enabled")
    
    try:
        token = _auth_manager.generate_jwt(key_id, expires_minutes)
        return {"token": token, "expires_in_minutes": expires_minutes}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/alerts", tags=["Monitoring"])
async def get_alerts(level: Optional[str] = None):
    """Get active alerts."""
    agent = require_agent(_agent)
    
    from ..core.alert_manager import AlertLevel
    
    alert_level = None
    if level:
        try:
            alert_level = AlertLevel(level)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid alert level: {level}. Valid: {[e.value for e in AlertLevel]}"
            )
    
    alerts = agent.get_active_alerts(level=alert_level)
    
    return {
        "active_alerts": len(alerts),
        "alerts": [
            {
                "type": alert.type.value,
                "level": alert.level.value,
                "message": alert.message,
                "details": alert.details,
                "timestamp": alert.timestamp.isoformat(),
            }
            for alert in alerts
        ]
    }


@app.get("/alerts/history", tags=["Monitoring"])
async def get_alert_history(limit: int = 100):
    """Get alert history."""
    agent = require_agent(_agent)
    alerts = agent.get_alert_history(limit=limit)
    
    return {
        "total": len(alerts),
        "alerts": [
            {
                "type": alert.type.value,
                "level": alert.level.value,
                "message": alert.message,
                "details": alert.details,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
            }
            for alert in alerts
        ]
    }


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    agent = require_agent(_agent)
    
    from ..core.prometheus_metrics import PrometheusMetrics
    
    metrics = PrometheusMetrics()
    
    # Collect metrics from agent
    stats = agent.get_performance_stats()
    
    # Executor metrics
    executor_stats = stats.get("executor_stats", {})
    metrics.set_gauge("executor_total_tasks", executor_stats.get("total_tasks", 0))
    metrics.set_gauge("executor_completed_tasks", executor_stats.get("completed_tasks", 0))
    metrics.set_gauge("executor_failed_tasks", executor_stats.get("failed_tasks", 0))
    metrics.set_gauge("executor_active_workers", executor_stats.get("active_workers", 0))
    metrics.set_gauge("executor_success_rate", executor_stats.get("success_rate", 0))
    
    # Cache metrics
    cache_stats = stats.get("cache_stats", {})
    metrics.set_gauge("cache_hits", cache_stats.get("hits", 0))
    metrics.set_gauge("cache_misses", cache_stats.get("misses", 0))
    metrics.set_gauge("cache_hit_rate", cache_stats.get("hit_rate", 0))
    metrics.set_gauge("cache_size", cache_stats.get("cache_size", 0))
    
    # Memory metrics
    memory_usage = stats.get("memory_usage", {})
    metrics.set_gauge("memory_process_mb", memory_usage.get("process_memory_mb", 0))
    metrics.set_gauge("memory_process_percent", memory_usage.get("process_memory_percent", 0))
    metrics.set_gauge("memory_system_percent", memory_usage.get("system_memory_percent", 0))
    
    # Webhook metrics
    webhook_stats = stats.get("webhook_stats", {})
    metrics.set_gauge("webhook_total_sent", webhook_stats.get("total_sent", 0))
    metrics.set_gauge("webhook_successful", webhook_stats.get("successful", 0))
    metrics.set_gauge("webhook_failed", webhook_stats.get("failed", 0))
    
    return Response(
        content=metrics.export_prometheus(),
        media_type="text/plain"
    )

