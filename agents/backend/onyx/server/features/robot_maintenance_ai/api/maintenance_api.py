"""
FastAPI endpoints for Robot Maintenance AI.
Refactored to use centralized schemas, dependencies, error handling, and BaseRouter.
"""

from fastapi import Depends, Request
from fastapi.responses import FileResponse
from typing import Optional, Dict, Any
from datetime import datetime
import tempfile
import os

from ..core.maintenance_tutor import RobotMaintenanceTutor
from ..core.conversation_manager import ConversationManager
from ..config.maintenance_config import MaintenanceConfig
from ..utils.metrics import metrics_collector
from ..utils.export_utils import export_conversation_json, export_conversation_csv, generate_maintenance_report

# Import refactored modules
from .schemas import (
    MaintenanceQuestionRequest,
    ProcedureRequest,
    DiagnosisRequest,
    PredictionRequest,
    ChecklistRequest,
    ScheduleRequest
)
from .dependencies import (
    get_tutor,
    get_conversation_manager,
    get_rate_limiter,
    check_rate_limit
)
from .exceptions import NotFoundError
from .base_router import BaseRouter

# Create base router instance
base = BaseRouter(
    prefix="/api/robot-maintenance",
    tags=["Robot Maintenance AI"],
    require_authentication=False,  # Some endpoints don't require auth
    require_rate_limit=True  # Rate limiting enabled for main endpoints
)

router = base.router


@router.post("/ask")
@base.timed_endpoint("ask_maintenance_question")
async def ask_maintenance_question(
    request: MaintenanceQuestionRequest,
    http_request: Request,
    tutor: RobotMaintenanceTutor = Depends(get_tutor),
    conversation_manager: ConversationManager = Depends(get_conversation_manager),
    _: str = Depends(check_rate_limit)
) -> Dict[str, Any]:
    """Ask a maintenance question to the AI tutor."""
    base.log_request("ask_maintenance_question", robot_type=request.robot_type, has_conversation_id=bool(request.conversation_id))
    
    # Build context from conversation history if available
    context = None
    if request.conversation_id:
        context_messages = conversation_manager.get_context(request.conversation_id)
        if context_messages:
            context = "\n".join([msg["content"] for msg in context_messages])
        if request.context:
            context = f"{context}\n\n{request.context}" if context else request.context
    
    # Call tutor (errors handled by middleware)
    response = await tutor.ask_maintenance_question(
        question=request.question,
        robot_type=request.robot_type,
        maintenance_type=request.maintenance_type,
        context=context or request.context,
        sensor_data=request.sensor_data
    )
    
    # Save to conversation if conversation_id provided
    if request.conversation_id:
        conversation_manager.add_message(
            request.conversation_id,
            "user",
            request.question
        )
        conversation_manager.add_message(
            request.conversation_id,
            "assistant",
            response["answer"]
        )
    
    return base.success(response)


@router.post("/procedure")
@base.timed_endpoint("get_maintenance_procedure")
async def get_maintenance_procedure(
    request: ProcedureRequest,
    tutor: RobotMaintenanceTutor = Depends(get_tutor)
) -> Dict[str, Any]:
    """Get a detailed maintenance procedure."""
    base.log_request("get_maintenance_procedure", robot_type=request.robot_type, procedure=request.procedure)
    
    response = await tutor.explain_maintenance_procedure(
        procedure=request.procedure,
        robot_type=request.robot_type,
        difficulty=request.difficulty
    )
    return base.success(response)


@router.post("/diagnose")
@base.timed_endpoint("diagnose_problem")
async def diagnose_problem(
    request: DiagnosisRequest,
    tutor: RobotMaintenanceTutor = Depends(get_tutor)
) -> Dict[str, Any]:
    """Diagnose a robot/machine problem."""
    base.log_request("diagnose_problem", robot_type=request.robot_type, symptoms_count=len(request.symptoms))
    
    response = await tutor.diagnose_problem(
        symptoms=request.symptoms,
        robot_type=request.robot_type,
        sensor_data=request.sensor_data
    )
    return base.success(response)


@router.post("/predict")
@base.timed_endpoint("predict_maintenance")
async def predict_maintenance(
    request: PredictionRequest,
    tutor: RobotMaintenanceTutor = Depends(get_tutor)
) -> Dict[str, Any]:
    """Predict maintenance schedule using ML."""
    base.log_request("predict_maintenance", robot_type=request.robot_type)
    
    response = await tutor.predict_maintenance_schedule(
        robot_type=request.robot_type,
        sensor_data=request.sensor_data,
        historical_data=request.historical_data
    )
    return base.success(response)


@router.post("/checklist")
@base.timed_endpoint("generate_checklist")
async def generate_checklist(
    request: ChecklistRequest,
    tutor: RobotMaintenanceTutor = Depends(get_tutor)
) -> Dict[str, Any]:
    """Generate a maintenance checklist."""
    base.log_request("generate_checklist", robot_type=request.robot_type, maintenance_type=request.maintenance_type)
    
    response = await tutor.generate_maintenance_checklist(
        robot_type=request.robot_type,
        maintenance_type=request.maintenance_type
    )
    return base.success(response)


@router.get("/conversation/{conversation_id}")
@base.timed_endpoint("get_conversation")
async def get_conversation(
    conversation_id: str,
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
) -> Dict[str, Any]:
    """Get conversation history."""
    base.log_request("get_conversation", conversation_id=conversation_id)
    
    conversation = conversation_manager.get_conversation(conversation_id)
    summary = conversation_manager.get_summary(conversation_id)
    return base.success({
        "conversation_id": conversation_id,
        "messages": conversation,
        "summary": summary
    })


@router.delete("/conversation/{conversation_id}")
@base.timed_endpoint("clear_conversation")
async def clear_conversation(
    conversation_id: str,
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
) -> Dict[str, Any]:
    """Clear conversation history."""
    base.log_request("clear_conversation", conversation_id=conversation_id)
    
    conversation_manager.clear_conversation(conversation_id)
    return base.success(None, message="Conversation cleared")


@router.get("/conversation/{conversation_id}/export/json")
async def export_conversation_json_endpoint(
    conversation_id: str,
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
) -> FileResponse:
    """Export conversation as JSON file."""
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise NotFoundError("Conversation", conversation_id)
    
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.json',
        delete=False,
        encoding='utf-8'
    )
    temp_file.close()
    
    export_conversation_json(conversation, temp_file.name)
    
    return FileResponse(
        temp_file.name,
        media_type='application/json',
        filename=f"conversation_{conversation_id}.json",
        background=lambda: os.unlink(temp_file.name)
    )


@router.get("/conversation/{conversation_id}/export/csv")
async def export_conversation_csv_endpoint(
    conversation_id: str,
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
) -> FileResponse:
    """Export conversation as CSV file."""
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise NotFoundError("Conversation", conversation_id)
    
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.csv',
        delete=False,
        encoding='utf-8'
    )
    temp_file.close()
    
    export_conversation_csv(conversation, temp_file.name)
    
    return FileResponse(
        temp_file.name,
        media_type='text/csv',
        filename=f"conversation_{conversation_id}.csv",
        background=lambda: os.unlink(temp_file.name)
    )


@router.get("/conversation/{conversation_id}/report")
@base.timed_endpoint("get_conversation_report")
async def get_conversation_report(
    conversation_id: str,
    robot_type: Optional[str] = None,
    maintenance_type: Optional[str] = None,
    conversation_manager: ConversationManager = Depends(get_conversation_manager)
) -> Dict[str, Any]:
    """Generate a maintenance report from conversation."""
    base.log_request("get_conversation_report", conversation_id=conversation_id)
    
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise NotFoundError("Conversation", conversation_id)
    
    report = generate_maintenance_report(
        conversation,
        robot_type or "unknown",
        maintenance_type or "general"
    )
    
    return base.success(report)


@router.get("/robot-types")
@base.timed_endpoint("get_robot_types")
async def get_robot_types() -> Dict[str, Any]:
    """Get list of supported robot types."""
    base.log_request("get_robot_types")
    
    config = MaintenanceConfig()
    return base.success({
        "robot_types": config.robot_types,
        "maintenance_categories": config.maintenance_categories,
        "difficulty_levels": config.difficulty_levels
    })


@router.get("/health")
async def health_check(
    tutor: RobotMaintenanceTutor = Depends(get_tutor)
) -> Dict[str, Any]:
    """Health check endpoint with detailed system status."""
    import sys
    import platform
    
    health_status = {
        "status": "healthy",
        "service": "Robot Maintenance AI Trainer",
        "version": "1.4.0",
        "timestamp": get_iso_timestamp(),
        "system": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "platform_version": platform.version()
        },
        "components": {
            "openrouter": {
                "configured": bool(tutor.config.openrouter.api_key),
                "base_url": tutor.config.openrouter.base_url,
                "model": tutor.config.openrouter.default_model
            },
            "nlp": {
                "enabled": tutor.config.nlp.use_spacy or tutor.config.nlp.use_transformers,
                "language": tutor.config.nlp.language
            },
            "ml": {
                "enabled": tutor.config.ml.enable_predictive_maintenance,
                "anomaly_detection": tutor.config.ml.enable_anomaly_detection
            },
            "cache": {
                "enabled": tutor.cache is not None,
                "size": tutor.cache.size() if tutor.cache else 0
            }
        },
        "metrics": {
            "uptime_seconds": metrics_collector.get_stats().get("uptime_seconds", 0),
            "total_requests": metrics_collector.get_stats().get("total_requests", 0)
        }
    }
    
    return health_status


@router.get("/cache/stats")
@base.timed_endpoint("get_cache_stats")
async def get_cache_stats(
    tutor: RobotMaintenanceTutor = Depends(get_tutor)
) -> Dict[str, Any]:
    """Get cache statistics."""
    base.log_request("get_cache_stats")
    
    if not tutor.cache:
        return base.success({
            "cache_enabled": False,
            "message": "Cache is disabled"
        })
    
    stats = tutor.cache.get_stats()
    return base.success({
        "cache_enabled": True,
        **stats
    })


@router.get("/metrics")
@base.timed_endpoint("get_metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get system metrics and statistics."""
    base.log_request("get_metrics")
    
    stats = metrics_collector.get_stats()
    return base.success(stats)


@router.post("/metrics/reset")
@base.timed_endpoint("reset_metrics")
async def reset_metrics() -> Dict[str, Any]:
    """Reset all metrics."""
    base.log_request("reset_metrics")
    
    metrics_collector.reset()
    return base.success(None, message="Metrics reset successfully")


@router.get("/rate-limit/stats")
@base.timed_endpoint("get_rate_limit_stats")
async def get_rate_limit_stats(
    rate_limiter = Depends(get_rate_limiter)
) -> Dict[str, Any]:
    """Get rate limiting statistics."""
    base.log_request("get_rate_limit_stats")
    
    stats = rate_limiter.get_stats()
    return base.success(stats)


@router.post("/rate-limit/reset")
@base.timed_endpoint("reset_rate_limit")
async def reset_rate_limit(
    http_request: Request,
    rate_limiter = Depends(get_rate_limiter)
) -> Dict[str, Any]:
    """Reset rate limit for current client."""
    base.log_request("reset_rate_limit")
    
    from .dependencies import get_client_identifier
    identifier = get_client_identifier(http_request)
    rate_limiter.reset(identifier)
    return base.success(None, message=f"Rate limit reset for {identifier}")


@router.delete("/cache/clear")
@base.timed_endpoint("clear_cache")
async def clear_cache(
    tutor: RobotMaintenanceTutor = Depends(get_tutor)
) -> Dict[str, Any]:
    """Clear the cache."""
    base.log_request("clear_cache")
    
    if not tutor.cache:
        return base.success(None, message="Cache is disabled, nothing to clear")
    
    tutor.cache.clear()
    return base.success(None, message="Cache cleared successfully")


def create_maintenance_app():
    """Create FastAPI app with maintenance routes."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from ..middleware.request_logging import RequestLoggingMiddleware
    from ..middleware.error_handler import ErrorHandlerMiddleware
    from .websocket_api import router as websocket_router
    from .auth_api import router as auth_router
    from .notifications_api import router as notifications_router
    from .admin_api import router as admin_router
    
    app = FastAPI(
        title="Robot Maintenance AI",
        version="2.2.0",
        description="AI system for teaching robot and machine maintenance using OpenRouter, NLP, and ML",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add error handling middleware first (outermost)
    app.add_middleware(ErrorHandlerMiddleware)
    
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Import new routers
    from .analytics_api import router as analytics_router
    from .search_api import router as search_router
    from .batch_api import router as batch_router
    from .plugins_api import router as plugins_router
    from .alerts_api import router as alerts_router
    from .recommendations_api import router as recommendations_router
    from .incidents_api import router as incidents_router
    from .comparison_api import router as comparison_router
    from .reports_api import router as reports_router
    from .learning_api import router as learning_router
    from .dashboard_api import router as dashboard_router
    from .webhooks_api import router as webhooks_router
    from .export_advanced_api import router as export_advanced_router
    from .config_api import router as config_router
    from .monitoring_api import router as monitoring_router
    from .audit_api import router as audit_router
    from .templates_api import router as templates_router
    from .validation_api import router as validation_router
    
    app.include_router(router)
    app.include_router(websocket_router)
    app.include_router(auth_router)
    app.include_router(notifications_router)
    app.include_router(admin_router)
    app.include_router(analytics_router)
    app.include_router(search_router)
    app.include_router(batch_router)
    app.include_router(plugins_router)
    app.include_router(alerts_router)
    app.include_router(recommendations_router)
    app.include_router(incidents_router)
    app.include_router(comparison_router)
    app.include_router(reports_router)
    app.include_router(learning_router)
    app.include_router(dashboard_router)
    app.include_router(webhooks_router)
    app.include_router(export_advanced_router)
    app.include_router(config_router)
    app.include_router(monitoring_router)
    app.include_router(audit_router)
    app.include_router(templates_router)
    app.include_router(validation_router)
    return app
