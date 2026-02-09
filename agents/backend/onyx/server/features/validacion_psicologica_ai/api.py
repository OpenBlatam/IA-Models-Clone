"""
API Endpoints para Validación Psicológica AI
=============================================
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.db.engine import get_session
from onyx.db.models import User

from .models import (
    SocialMediaPlatform,
    PsychologicalValidation,
    SocialMediaConnection,
)
from .schemas import (
    ValidationCreate,
    ValidationRead,
    SocialMediaConnectRequest,
    SocialMediaConnectionResponse,
    ValidationDetailResponse,
    ValidationReportResponse,
    PsychologicalProfileResponse,
    SocialMediaDataRequest,
    SocialMediaDataResponse,
)
from .service import PsychologicalValidationService
from .exporters import ReportExporter, PDFExporter
from .utils import ValidationComparator, metrics, CacheManager
from .alerts import alert_manager
from .recommendations import recommendation_engine
from .webhooks import webhook_manager, WebhookEvent
from .predictive_analysis import predictive_analyzer
from .dashboard import dashboard_generator
from .versioning import version_manager
from .feedback import feedback_manager, FeedbackType, FeedbackRating
from .batch_processor import batch_processor
from .health import health_checker
from .notifications import notification_manager, NotificationType, NotificationPriority
from .backup import backup_manager
from .rate_limiter import rate_limiter
from .audit import audit_logger, AuditAction
from .integrations import integration_manager
from .permissions import permission_manager, Permission, Role
from .quotas import quota_manager, QuotaType
from .comparative_analysis import comparative_analyzer
from .report_templates import template_manager, TemplateType
from .ai_integrations import ai_service_manager
from .queue_system import queue_manager, JobPriority
from .distributed_cache import distributed_cache
from .i18n import translator, Language
from .ab_testing import ab_test_manager, Variant, ExperimentStatus
from .metrics import metrics_collector
from .event_bus import event_bus, EventType
from .api_versioning import api_version_manager, APIVersion
from .migrations import migration_manager
from .data_validation import data_validator
from .data_transformation import data_transformer
from .sync import sync_manager
from .deep_learning_models import deep_learning_analyzer
from .fine_tuning import lora_trainer
from .diffusion_models import visualization_generator
from .gradio_interface import gradio_interface
from .experiment_tracking import experiment_tracker
from .evaluation_module import ModelEvaluator
from .checkpointing import checkpoint_manager
from .inference_engine import InferenceEngine, ModelServer
from .profiling import performance_profiler, ModelOptimizer
from .debugging_tools import model_debugger, DataDebugger
from .data_augmentation import text_augmenter, AugmentedDataset
from .ensemble_models import ModelEnsemble, StackingEnsemble
from .transfer_learning import transfer_learning_manager
from .hyperparameter_tuning import hyperparameter_tuner, HyperparameterSpace, LearningRateFinder
from .model_export import model_exporter, ModelLoader
from .memory_optimization import memory_optimizer, BatchMemoryManager
from .diffusion_improvements import advanced_diffusion_pipeline, DiffusionImageEnhancer
from .model_architecture import ImprovedPersonalityModel
from .validation_utils import ModelValidator, GradientValidator
from .experiment_manager import experiment_manager
from .advanced_logging import advanced_logger
from .model_serving import model_registry
from .monitoring import system_monitor, health_checker
from .model_optimization import model_optimizer, model_quantizer, model_pruner
from .deployment import model_deployment, model_versioning
from .benchmarking import benchmark_suite, model_benchmark
from .model_security import model_security, model_sanitizer
from fastapi.responses import Response, StreamingResponse, FileResponse
from fastapi import WebSocket, WebSocketDisconnect, Request, HTTPException as FastAPIHTTPException
from PIL import Image
import io

import structlog

logger = structlog.get_logger()
router = APIRouter(prefix="/psychological-validation", tags=["psychological-validation"])

# Instancia global del servicio (en producción, usar inyección de dependencias)
_service = PsychologicalValidationService()


@router.post("/connect", response_model=SocialMediaConnectionResponse)
async def connect_social_media(
    request: SocialMediaConnectRequest,
    user: User = Depends(current_user),
) -> SocialMediaConnectionResponse:
    """
    Conectar una red social del usuario
    
    Permite conectar una cuenta de red social para su análisis psicológico.
    """
    try:
        connection = await _service.connect_social_media(
            user_id=user.id,
            request=request
        )
        
        return SocialMediaConnectionResponse(
            id=connection.id,
            platform=connection.platform,
            status=connection.status,
            connected_at=connection.connected_at,
            profile_data=connection.profile_data
        )
    except Exception as e:
        logger.error("Error connecting social media", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/connect/{platform}")
async def disconnect_social_media(
    platform: SocialMediaPlatform = Path(..., description="Plataforma a desconectar"),
    user: User = Depends(current_user),
) -> dict:
    """
    Desconectar una red social del usuario
    """
    try:
        success = await _service.disconnect_social_media(
            user_id=user.id,
            platform=platform
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Connection not found")
        
        return {"message": "Social media disconnected successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error disconnecting social media", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connections", response_model=List[SocialMediaConnectionResponse])
async def get_connections(
    user: User = Depends(current_user),
) -> List[SocialMediaConnectionResponse]:
    """
    Obtener todas las conexiones de redes sociales del usuario
    """
    try:
        connections = await _service.get_user_connections(user.id)
        
        return [
            SocialMediaConnectionResponse(
                id=conn.id,
                platform=conn.platform,
                status=conn.status,
                connected_at=conn.connected_at,
                profile_data=conn.profile_data
            )
            for conn in connections
        ]
    except Exception as e:
        logger.error("Error getting connections", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validations", response_model=ValidationRead)
async def create_validation(
    request: ValidationCreate,
    user: User = Depends(current_user),
    http_request: Request = None,
) -> ValidationRead:
    """
    Crear una nueva validación psicológica
    
    Inicia el proceso de análisis psicológico basado en las redes sociales conectadas.
    """
    try:
        validation = await _service.create_validation(
            user_id=user.id,
            request=request
        )
        
        # Registrar en auditoría
        try:
            from .audit import audit_logger, AuditAction
            audit_logger.log(
                user_id=user.id,
                action=AuditAction.VALIDATION_CREATED,
                resource_id=validation.id,
                resource_type="validation",
                ip_address=http_request.client.host if http_request and http_request.client else None,
                user_agent=http_request.headers.get("user-agent") if http_request else None
            )
        except Exception as e:
            logger.warning("Error logging audit", error=str(e))
        
        return ValidationRead(
            id=validation.id,
            user_id=validation.user_id,
            status=validation.status,
            connected_platforms=validation.connected_platforms,
            created_at=validation.created_at,
            updated_at=validation.updated_at,
            completed_at=validation.completed_at,
            has_profile=validation.profile is not None,
            has_report=validation.report is not None
        )
    except Exception as e:
        logger.error("Error creating validation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validations/{validation_id}/run", response_model=ValidationDetailResponse)
async def run_validation(
    validation_id: UUID = Path(..., description="ID de la validación"),
    user: User = Depends(current_user),
) -> ValidationDetailResponse:
    """
    Ejecutar análisis de validación psicológica
    
    Procesa los datos de redes sociales y genera el perfil psicológico y reporte.
    """
    try:
        validation = await _service.run_validation(validation_id)
        
        # Verificar que la validación pertenece al usuario
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return ValidationDetailResponse(
            validation=ValidationRead(
                id=validation.id,
                user_id=validation.user_id,
                status=validation.status,
                connected_platforms=validation.connected_platforms,
                created_at=validation.created_at,
                updated_at=validation.updated_at,
                completed_at=validation.completed_at,
                has_profile=validation.profile is not None,
                has_report=validation.report is not None
            ),
            profile=PsychologicalProfileResponse(
                id=validation.profile.id,
                user_id=validation.profile.user_id,
                personality_traits=validation.profile.personality_traits,
                emotional_state=validation.profile.emotional_state,
                behavioral_patterns=validation.profile.behavioral_patterns,
                risk_factors=validation.profile.risk_factors,
                strengths=validation.profile.strengths,
                recommendations=validation.profile.recommendations,
                confidence_score=validation.profile.confidence_score,
                created_at=validation.profile.created_at,
                updated_at=validation.profile.updated_at
            ) if validation.profile else None,
            report=ValidationReportResponse(
                id=validation.report.id,
                validation_id=validation.report.validation_id,
                summary=validation.report.summary,
                detailed_analysis=validation.report.detailed_analysis,
                social_media_insights=validation.report.social_media_insights,
                timeline_analysis=validation.report.timeline_analysis,
                sentiment_analysis=validation.report.sentiment_analysis,
                content_analysis=validation.report.content_analysis,
                interaction_patterns=validation.report.interaction_patterns,
                generated_at=validation.report.generated_at
            ) if validation.report else None,
            connections=await _service.get_user_connections(user.id)
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Error running validation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations", response_model=List[ValidationRead])
async def get_validations(
    user: User = Depends(current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
) -> List[ValidationRead]:
    """
    Obtener todas las validaciones del usuario
    """
    try:
        validations = await _service.get_user_validations(user.id)
        
        # Aplicar paginación
        paginated = validations[skip:skip + limit]
        
        return [
            ValidationRead(
                id=v.id,
                user_id=v.user_id,
                status=v.status,
                connected_platforms=v.connected_platforms,
                created_at=v.created_at,
                updated_at=v.updated_at,
                completed_at=v.completed_at,
                has_profile=v.profile is not None,
                has_report=v.report is not None
            )
            for v in paginated
        ]
    except Exception as e:
        logger.error("Error getting validations", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations/{validation_id}", response_model=ValidationDetailResponse)
async def get_validation(
    validation_id: UUID = Path(..., description="ID de la validación"),
    user: User = Depends(current_user),
) -> ValidationDetailResponse:
    """
    Obtener detalles de una validación específica
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return ValidationDetailResponse(
            validation=ValidationRead(
                id=validation.id,
                user_id=validation.user_id,
                status=validation.status,
                connected_platforms=validation.connected_platforms,
                created_at=validation.created_at,
                updated_at=validation.updated_at,
                completed_at=validation.completed_at,
                has_profile=validation.profile is not None,
                has_report=validation.report is not None
            ),
            profile=PsychologicalProfileResponse(
                id=validation.profile.id,
                user_id=validation.profile.user_id,
                personality_traits=validation.profile.personality_traits,
                emotional_state=validation.profile.emotional_state,
                behavioral_patterns=validation.profile.behavioral_patterns,
                risk_factors=validation.profile.risk_factors,
                strengths=validation.profile.strengths,
                recommendations=validation.profile.recommendations,
                confidence_score=validation.profile.confidence_score,
                created_at=validation.profile.created_at,
                updated_at=validation.profile.updated_at
            ) if validation.profile else None,
            report=ValidationReportResponse(
                id=validation.report.id,
                validation_id=validation.report.validation_id,
                summary=validation.report.summary,
                detailed_analysis=validation.report.detailed_analysis,
                social_media_insights=validation.report.social_media_insights,
                timeline_analysis=validation.report.timeline_analysis,
                sentiment_analysis=validation.report.sentiment_analysis,
                content_analysis=validation.report.content_analysis,
                interaction_patterns=validation.report.interaction_patterns,
                generated_at=validation.report.generated_at
            ) if validation.report else None,
            connections=await _service.get_user_connections(user.id)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting validation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profile/{validation_id}", response_model=PsychologicalProfileResponse)
async def get_profile(
    validation_id: UUID = Path(..., description="ID de la validación"),
    user: User = Depends(current_user),
) -> PsychologicalProfileResponse:
    """
    Obtener perfil psicológico de una validación
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not validation.profile:
            raise HTTPException(status_code=404, detail="Profile not generated yet")
        
        return PsychologicalProfileResponse(
            id=validation.profile.id,
            user_id=validation.profile.user_id,
            personality_traits=validation.profile.personality_traits,
            emotional_state=validation.profile.emotional_state,
            behavioral_patterns=validation.profile.behavioral_patterns,
            risk_factors=validation.profile.risk_factors,
            strengths=validation.profile.strengths,
            recommendations=validation.profile.recommendations,
            confidence_score=validation.profile.confidence_score,
            created_at=validation.profile.created_at,
            updated_at=validation.profile.updated_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting profile", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report/{validation_id}", response_model=ValidationReportResponse)
async def get_report(
    validation_id: UUID = Path(..., description="ID de la validación"),
    user: User = Depends(current_user),
) -> ValidationReportResponse:
    """
    Obtener reporte de validación
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not validation.report:
            raise HTTPException(status_code=404, detail="Report not generated yet")
        
        return ValidationReportResponse(
            id=validation.report.id,
            validation_id=validation.report.validation_id,
            summary=validation.report.summary,
            detailed_analysis=validation.report.detailed_analysis,
            social_media_insights=validation.report.social_media_insights,
            timeline_analysis=validation.report.timeline_analysis,
            sentiment_analysis=validation.report.sentiment_analysis,
            content_analysis=validation.report.content_analysis,
            interaction_patterns=validation.report.interaction_patterns,
            generated_at=validation.report.generated_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting report", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations/{validation_id}/export/json")
async def export_validation_json(
    validation_id: UUID = Path(..., description="ID de la validación"),
    user: User = Depends(current_user),
) -> Response:
    """
    Exportar validación en formato JSON
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        json_data = ReportExporter.export_to_json(
            validation.report,
            validation.profile
        )
        
        return Response(
            content=json_data,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=validation_{validation_id}.json"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error exporting JSON", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations/{validation_id}/export/text")
async def export_validation_text(
    validation_id: UUID = Path(..., description="ID de la validación"),
    user: User = Depends(current_user),
) -> Response:
    """
    Exportar validación en formato texto
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        text_data = ReportExporter.export_to_text(
            validation.report,
            validation.profile
        )
        
        return Response(
            content=text_data,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=validation_{validation_id}.txt"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error exporting text", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations/{validation_id}/export/html")
async def export_validation_html(
    validation_id: UUID = Path(..., description="ID de la validación"),
    user: User = Depends(current_user),
) -> Response:
    """
    Exportar validación en formato HTML
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        html_data = ReportExporter.export_to_html(
            validation.report,
            validation.profile
        )
        
        return Response(
            content=html_data,
            media_type="text/html",
            headers={
                "Content-Disposition": f"attachment; filename=validation_{validation_id}.html"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error exporting HTML", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations/{validation_id}/export/pdf")
async def export_validation_pdf(
    validation_id: UUID = Path(..., description="ID de la validación"),
    user: User = Depends(current_user),
) -> StreamingResponse:
    """
    Exportar validación en formato PDF
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not validation.report:
            raise HTTPException(status_code=404, detail="Report not generated yet")
        
        pdf_buffer = PDFExporter.export_to_pdf(
            validation.report,
            validation.profile
        )
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=validation_{validation_id}.pdf"
            }
        )
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="PDF export requires reportlab. Install with: pip install reportlab"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error exporting PDF", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations/compare")
async def compare_validations(
    validation_id1: UUID = Query(..., description="ID de la primera validación"),
    validation_id2: UUID = Query(..., description="ID de la segunda validación"),
    user: User = Depends(current_user),
) -> dict:
    """
    Comparar dos validaciones
    """
    try:
        validation1 = await _service.get_validation(validation_id1)
        validation2 = await _service.get_validation(validation_id2)
        
        if not validation1 or not validation2:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation1.user_id != user.id or validation2.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not validation1.profile or not validation2.profile:
            raise HTTPException(status_code=400, detail="Profiles not available for comparison")
        
        comparison = ValidationComparator.compare_validations(
            validation1.to_dict(),
            validation2.to_dict()
        )
        
        return comparison
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error comparing validations", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts(
    user: User = Depends(current_user),
    alert_type: Optional[str] = Query(None, description="Filtrar por tipo"),
    severity: Optional[str] = Query(None, description="Filtrar por severidad"),
    limit: int = Query(100, ge=1, le=1000),
) -> dict:
    """
    Obtener alertas del usuario
    """
    try:
        from .alerts import AlertType, AlertSeverity
        
        alert_type_enum = AlertType(alert_type) if alert_type else None
        severity_enum = AlertSeverity(severity) if severity else None
        
        alerts = alert_manager.get_alerts(
            alert_type=alert_type_enum,
            severity=severity_enum,
            limit=limit
        )
        
        return {
            "alerts": [alert.to_dict() for alert in alerts],
            "total": len(alerts),
            "summary": alert_manager.get_alerts_summary()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filter: {str(e)}")
    except Exception as e:
        logger.error("Error getting alerts", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics(
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener métricas del sistema
    """
    try:
        all_metrics = metrics.get_all()
        return all_metrics
    except Exception as e:
        logger.error("Error getting metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations/{validation_id}/recommendations")
async def get_recommendations(
    validation_id: UUID = Path(..., description="ID de la validación"),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener recomendaciones personalizadas para una validación
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not validation.profile:
            raise HTTPException(status_code=404, detail="Profile not generated yet")
        
        recommendations = recommendation_engine.generate_recommendations(
            validation.profile,
            validation.report
        )
        
        return {
            "recommendations": [r.to_dict() for r in recommendations],
            "total": len(recommendations),
            "validation_id": str(validation_id)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting recommendations", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations/{validation_id}/predictions")
async def get_predictions(
    validation_id: UUID = Path(..., description="ID de la validación"),
    timeframe_days: int = Query(30, ge=7, le=365, description="Período de predicción en días"),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener predicciones basadas en análisis histórico
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Obtener validaciones históricas
        historical_validations = await _service.get_user_validations(user.id)
        historical_profiles = [
            v.profile for v in historical_validations
            if v.profile and v.id != validation_id
        ]
        
        if len(historical_profiles) < 2:
            raise HTTPException(
                status_code=400,
                detail="Insufficient historical data for predictions. Need at least 2 previous validations."
            )
        
        predictions = predictive_analyzer.analyze_trends(
            historical_profiles,
            timeframe_days
        )
        
        # Detectar anomalías
        if validation.profile:
            anomalies = predictive_analyzer.detect_anomalies(
                validation.profile,
                historical_profiles
            )
        else:
            anomalies = []
        
        return {
            "predictions": {k: v.to_dict() for k, v in predictions.items()},
            "anomalies": anomalies,
            "timeframe_days": timeframe_days,
            "historical_data_points": len(historical_profiles)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting predictions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhooks")
async def register_webhook(
    url: str = Query(..., description="URL del webhook"),
    events: List[str] = Query(..., description="Eventos a escuchar"),
    secret: Optional[str] = Query(None, description="Secreto para validación"),
    user: User = Depends(current_user),
) -> dict:
    """
    Registrar un nuevo webhook
    """
    try:
        from .webhooks import WebhookEvent
        
        webhook_events = []
        for event_str in events:
            try:
                webhook_events.append(WebhookEvent(event_str))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid event: {event_str}"
                )
        
        webhook = webhook_manager.register_webhook(url, webhook_events, secret)
        
        return {
            "webhook": webhook.to_dict(),
            "message": "Webhook registered successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error registering webhook", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/webhooks")
async def get_webhooks(
    user: User = Depends(current_user),
    active_only: bool = Query(False, description="Solo webhooks activos"),
) -> dict:
    """
    Obtener webhooks registrados
    """
    try:
        webhooks = webhook_manager.get_webhooks(active_only=active_only)
        return {
            "webhooks": [w.to_dict() for w in webhooks],
            "total": len(webhooks)
        }
    except Exception as e:
        logger.error("Error getting webhooks", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: UUID = Path(..., description="ID del webhook"),
    user: User = Depends(current_user),
) -> dict:
    """
    Eliminar un webhook
    """
    try:
        success = webhook_manager.unregister_webhook(webhook_id)
        if not success:
            raise HTTPException(status_code=404, detail="Webhook not found")
        
        return {"message": "Webhook deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting webhook", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_dashboard(
    user: User = Depends(current_user),
    days: int = Query(30, ge=7, le=365, description="Días de datos históricos"),
) -> dict:
    """
    Obtener datos completos del dashboard
    """
    try:
        validations = await _service.get_user_validations(user.id)
        dashboard_data = dashboard_generator.generate_complete_dashboard(validations)
        return dashboard_data
    except Exception as e:
        logger.error("Error generating dashboard", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations/{validation_id}/versions")
async def get_validation_versions(
    validation_id: UUID = Path(..., description="ID de la validación"),
    include_deprecated: bool = Query(False, description="Incluir versiones deprecadas"),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener versiones de una validación
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        versions = version_manager.get_versions(validation_id, include_deprecated)
        
        return {
            "validation_id": str(validation_id),
            "versions": [v.to_dict() for v in versions],
            "total": len(versions)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting versions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations/{validation_id}/versions/{version_number}")
async def get_version(
    validation_id: UUID = Path(..., description="ID de la validación"),
    version_number: int = Path(..., description="Número de versión"),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener una versión específica
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        versions = version_manager.get_versions(validation_id, include_deprecated=True)
        target_version = next(
            (v for v in versions if v.version_number == version_number),
            None
        )
        
        if not target_version:
            raise HTTPException(status_code=404, detail="Version not found")
        
        return {
            "version": target_version.to_dict(),
            "validation_data": target_version.validation_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting version", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validations/{validation_id}/versions/compare")
async def compare_versions(
    validation_id: UUID = Path(..., description="ID de la validación"),
    version1: int = Query(..., description="Primera versión"),
    version2: int = Query(..., description="Segunda versión"),
    user: User = Depends(current_user),
) -> dict:
    """
    Comparar dos versiones de una validación
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        comparison = version_manager.compare_versions(validation_id, version1, version2)
        
        return comparison
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error comparing versions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validations/{validation_id}/feedback")
async def submit_feedback(
    validation_id: UUID = Path(..., description="ID de la validación"),
    feedback_type: FeedbackType = Query(..., description="Tipo de feedback"),
    rating: FeedbackRating = Query(..., description="Calificación"),
    comment: Optional[str] = Query(None, description="Comentario"),
    user: User = Depends(current_user),
) -> dict:
    """
    Enviar feedback sobre una validación
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        feedback = feedback_manager.submit_feedback(
            validation_id=validation_id,
            user_id=user.id,
            feedback_type=feedback_type,
            rating=rating,
            comment=comment
        )
        
        return {
            "feedback": feedback.to_dict(),
            "message": "Feedback submitted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error submitting feedback", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations/{validation_id}/feedback")
async def get_feedback(
    validation_id: UUID = Path(..., description="ID de la validación"),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener feedback de una validación
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        feedback = feedback_manager.get_feedback(validation_id=validation_id)
        stats = feedback_manager.get_feedback_stats(validation_id=validation_id)
        
        return {
            "feedback": [f.to_dict() for f in feedback],
            "stats": stats,
            "total": len(feedback)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting feedback", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/process")
async def process_batch(
    validation_ids: List[UUID] = Query(..., description="IDs de validaciones a procesar"),
    user: User = Depends(current_user),
) -> dict:
    """
    Procesar lote de validaciones
    """
    try:
        # Verificar que todas las validaciones pertenecen al usuario
        for val_id in validation_ids:
            validation = await _service.get_validation(val_id)
            if not validation or validation.user_id != user.id:
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied for validation {val_id}"
                )
        
        if not batch_processor:
            raise HTTPException(
                status_code=503,
                detail="Batch processing not available"
            )
        
        job = await batch_processor.process_batch(validation_ids)
        
        return {
            "job": {
                "id": str(job.id),
                "status": job.status,
                "total": len(job.validation_ids),
                "successful": len(job.results),
                "failed": len(job.errors)
            },
            "message": "Batch processing started"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing batch", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch/jobs/{job_id}")
async def get_batch_job(
    job_id: UUID = Path(..., description="ID del trabajo"),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener estado de un trabajo de procesamiento por lotes
    """
    try:
        if not batch_processor:
            raise HTTPException(
                status_code=503,
                detail="Batch processing not available"
            )
        
        job = batch_processor.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        stats = batch_processor.get_job_stats(job_id)
        
        return {
            "job": job.__dict__ if hasattr(job, '__dict__') else {},
            "stats": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting batch job", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> dict:
    """
    Health check del sistema
    """
    try:
        if not health_checker:
            return {
                "status": "unknown",
                "message": "Health checker not initialized"
            }
        
        health_status = await health_checker.check_health()
        return health_status
    except Exception as e:
        logger.error("Error checking health", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/notifications")
async def get_notifications(
    user: User = Depends(current_user),
    unread_only: bool = Query(False, description="Solo no leídas"),
    limit: int = Query(100, ge=1, le=1000),
) -> dict:
    """
    Obtener notificaciones del usuario
    """
    try:
        notifications = notification_manager.get_notifications(
            user.id,
            unread_only=unread_only,
            limit=limit
        )
        unread_count = notification_manager.get_unread_count(user.id)
        
        return {
            "notifications": [n.to_dict() for n in notifications],
            "unread_count": unread_count,
            "total": len(notifications)
        }
    except Exception as e:
        logger.error("Error getting notifications", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: UUID = Path(..., description="ID de la notificación"),
    user: User = Depends(current_user),
) -> dict:
    """
    Marcar notificación como leída
    """
    try:
        success = notification_manager.mark_as_read(user.id, notification_id)
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return {"message": "Notification marked as read"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error marking notification as read", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/notifications/read-all")
async def mark_all_notifications_read(
    user: User = Depends(current_user),
) -> dict:
    """
    Marcar todas las notificaciones como leídas
    """
    try:
        count = notification_manager.mark_all_as_read(user.id)
        return {
            "message": "All notifications marked as read",
            "count": count
        }
    except Exception as e:
        logger.error("Error marking all notifications as read", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/notifications/{notification_id}")
async def delete_notification(
    notification_id: UUID = Path(..., description="ID de la notificación"),
    user: User = Depends(current_user),
) -> dict:
    """
    Eliminar notificación
    """
    try:
        success = notification_manager.delete_notification(user.id, notification_id)
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return {"message": "Notification deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting notification", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/notifications/ws")
async def notifications_websocket(websocket: WebSocket):
    """
    WebSocket para notificaciones en tiempo real
    """
    await websocket.accept()
    user_id = None
    
    try:
        # En producción, autenticar usuario desde token
        # Por ahora, usar query param
        user_id_str = websocket.query_params.get("user_id")
        if not user_id_str:
            await websocket.close(code=1008, reason="User ID required")
            return
        
        user_id = UUID(user_id_str)
        
        # Callback para enviar notificaciones
        async def send_notification(notification):
            try:
                await websocket.send_json(notification.to_dict())
            except Exception as e:
                logger.error("Error sending notification via WebSocket", error=str(e))
        
        # Suscribirse a notificaciones
        notification_manager.subscribe(user_id, send_notification)
        
        # Mantener conexión abierta
        while True:
            try:
                # Esperar mensajes del cliente (ping/pong)
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error("WebSocket error", error=str(e))
                break
    
    finally:
        if user_id:
            notification_manager.unsubscribe(user_id, send_notification)
        await websocket.close()


@router.post("/backup/create")
async def create_backup(
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> dict:
    """
    Crear backup de datos del usuario
    """
    try:
        # Obtener todas las validaciones y conexiones del usuario
        validations = await _service.get_user_validations(user.id)
        connections = await _service.get_user_connections(user.id)
        
        backup_info = backup_manager.create_backup(
            validations=validations,
            connections=connections,
            metadata={"user_id": str(user.id), "created_by": user.email if hasattr(user, 'email') else None}
        )
        
        # Registrar en auditoría
        audit_logger.log(
            user_id=user.id,
            action=AuditAction.SETTINGS_CHANGED,
            resource_type="backup",
            details={"backup_id": backup_info["backup_id"]}
        )
        
        return {
            "backup": backup_info,
            "message": "Backup created successfully"
        }
    except Exception as e:
        logger.error("Error creating backup", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backup/list")
async def list_backups(
    user: User = Depends(current_user),
    limit: int = Query(100, ge=1, le=1000),
) -> dict:
    """
    Listar backups disponibles
    """
    try:
        backups = backup_manager.list_backups(limit=limit)
        return {
            "backups": backups,
            "total": len(backups)
        }
    except Exception as e:
        logger.error("Error listing backups", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backup/{backup_id}/restore")
async def restore_backup(
    backup_id: str = Path(..., description="ID del backup"),
    user: User = Depends(current_user),
) -> dict:
    """
    Restaurar desde backup
    """
    try:
        backup_data = backup_manager.restore_backup(backup_id)
        
        # Registrar en auditoría
        audit_logger.log(
            user_id=user.id,
            action=AuditAction.SETTINGS_CHANGED,
            resource_type="backup",
            details={"backup_id": backup_id, "action": "restore"}
        )
        
        return {
            "backup": backup_data,
            "message": "Backup restored successfully"
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Backup not found")
    except Exception as e:
        logger.error("Error restoring backup", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/logs")
async def get_audit_logs(
    user: User = Depends(current_user),
    action: Optional[str] = Query(None, description="Filtrar por acción"),
    resource_type: Optional[str] = Query(None, description="Filtrar por tipo de recurso"),
    limit: int = Query(1000, ge=1, le=10000),
) -> dict:
    """
    Obtener logs de auditoría
    """
    try:
        from .audit import AuditAction
        
        action_enum = AuditAction(action) if action else None
        
        logs = audit_logger.get_logs(
            user_id=user.id,
            action=action_enum,
            resource_type=resource_type,
            limit=limit
        )
        
        return {
            "logs": [log.to_dict() for log in logs],
            "total": len(logs)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {str(e)}")
    except Exception as e:
        logger.error("Error getting audit logs", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/summary")
async def get_audit_summary(
    user: User = Depends(current_user),
    days: int = Query(30, ge=1, le=365),
) -> dict:
    """
    Obtener resumen de auditoría
    """
    try:
        summary = audit_logger.get_audit_summary(
            user_id=user.id,
            days=days
        )
        return summary
    except Exception as e:
        logger.error("Error getting audit summary", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quotas")
async def get_quotas(
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener cuotas del usuario
    """
    try:
        quotas = quota_manager.get_all_quotas(user.id)
        return quotas
    except Exception as e:
        logger.error("Error getting quotas", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quotas/{quota_type}")
async def get_quota_info(
    quota_type: QuotaType = Path(..., description="Tipo de cuota"),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener información de una cuota específica
    """
    try:
        info = quota_manager.get_usage_info(user.id, quota_type)
        return info
    except Exception as e:
        logger.error("Error getting quota info", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/compare")
async def compare_users(
    user_ids: List[UUID] = Query(..., description="IDs de usuarios a comparar"),
    user: User = Depends(current_user),
) -> dict:
    """
    Comparar múltiples usuarios
    """
    try:
        # Verificar permisos
        permission_manager.require_permission(user.id, Permission.VIEW_COMPARISONS)
        
        # Obtener perfiles de los usuarios
        profiles = []
        for user_id in user_ids:
            validations = await _service.get_user_validations(user_id)
            latest_validation = next(
                (v for v in validations if v.profile),
                None
            )
            if latest_validation and latest_validation.profile:
                profiles.append(latest_validation.profile)
        
        if len(profiles) < 2:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 users with profiles for comparison"
            )
        
        comparison = comparative_analyzer.compare_users(profiles, user_ids)
        
        return comparison
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error comparing users", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations/{validation_id}/benchmark")
async def benchmark_validation(
    validation_id: UUID = Path(..., description="ID de la validación"),
    user: User = Depends(current_user),
) -> dict:
    """
    Comparar validación contra población
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not validation.profile:
            raise HTTPException(status_code=404, detail="Profile not generated yet")
        
        # Obtener perfiles de población (simplificado - en producción usar datos reales)
        # Por ahora, usar otras validaciones del usuario como población
        all_validations = await _service.get_user_validations(user.id)
        population_profiles = [
            v.profile for v in all_validations
            if v.profile and v.id != validation_id
        ]
        
        benchmark = comparative_analyzer.benchmark_against_population(
            validation.profile,
            population_profiles
        )
        
        return benchmark
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error benchmarking validation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def get_templates(
    template_type: Optional[TemplateType] = Query(None, description="Filtrar por tipo"),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener plantillas de reportes disponibles
    """
    try:
        templates = template_manager.get_templates(template_type=template_type)
        return {
            "templates": [t.to_dict() for t in templates],
            "total": len(templates)
        }
    except Exception as e:
        logger.error("Error getting templates", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations/{validation_id}/report/template/{template_id}")
async def generate_report_from_template(
    validation_id: UUID = Path(..., description="ID de la validación"),
    template_id: UUID = Path(..., description="ID de la plantilla"),
    user: User = Depends(current_user),
) -> dict:
    """
    Generar reporte usando plantilla específica
    """
    try:
        validation = await _service.get_validation(validation_id)
        
        if not validation:
            raise HTTPException(status_code=404, detail="Validation not found")
        
        if validation.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not validation.report:
            raise HTTPException(status_code=404, detail="Report not generated yet")
        
        template = template_manager.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        generated_report = template_manager.generate_report_from_template(
            template,
            validation.report,
            validation.profile
        )
        
        return generated_report
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating report from template", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/permissions")
async def get_user_permissions(
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener permisos del usuario
    """
    try:
        permissions = permission_manager.get_user_permissions(user.id)
        roles = permission_manager.get_user_roles(user.id)
        
        return {
            "user_id": str(user.id),
            "roles": [r.value for r in roles],
            "permissions": [p.value for p in permissions],
            "total_permissions": len(permissions)
        }
    except Exception as e:
        logger.error("Error getting permissions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/queue/jobs")
async def enqueue_job(
    job_type: str = Query(..., description="Tipo de trabajo"),
    priority: JobPriority = Query(JobPriority.NORMAL, description="Prioridad"),
    data: Optional[Dict[str, Any]] = Query(None, description="Datos del trabajo"),
    user: User = Depends(current_user),
) -> dict:
    """
    Agregar trabajo a la cola
    """
    try:
        job_id = await queue_manager.enqueue(
            job_type=job_type,
            data=data or {},
            priority=priority
        )
        
        return {
            "job_id": str(job_id),
            "status": "queued",
            "message": "Job enqueued successfully"
        }
    except Exception as e:
        logger.error("Error enqueueing job", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue/jobs/{job_id}")
async def get_queue_job(
    job_id: UUID = Path(..., description="ID del trabajo"),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener estado de trabajo en cola
    """
    try:
        job = queue_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "job_id": str(job.id),
            "job_type": job.job_type,
            "status": job.status.value,
            "priority": job.priority.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "retry_count": job.retry_count,
            "error": job.error
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting queue job", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue/stats")
async def get_queue_stats(
    job_type: Optional[str] = Query(None, description="Filtrar por tipo"),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener estadísticas de cola
    """
    try:
        stats = queue_manager.get_queue_stats(job_type=job_type)
        return stats
    except Exception as e:
        logger.error("Error getting queue stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_stats(
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener estadísticas del caché distribuido
    """
    try:
        stats = await distributed_cache.get_stats()
        return stats
    except Exception as e:
        logger.error("Error getting cache stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai/analyze")
async def analyze_with_ai(
    text: str = Query(..., description="Texto a analizar"),
    task: str = Query("sentiment_analysis", description="Tarea de análisis"),
    service: Optional[str] = Query(None, description="Servicio de IA a usar"),
    user: User = Depends(current_user),
) -> dict:
    """
    Analizar texto usando servicios de IA externos
    """
    try:
        ai_service = ai_service_manager.get_service(service)
        if not ai_service:
            raise HTTPException(
                status_code=503,
                detail="No AI service available"
            )
        
        result = await ai_service.analyze_text(text, task)
        
        return {
            "result": result,
            "service": service or "default",
            "task": task
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error analyzing with AI", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/translations")
async def get_translations(
    language: Language = Query(Language.EN, description="Idioma"),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener traducciones para un idioma
    """
    try:
        translations = translator._translations.get(language, translator._translations[Language.EN])
        return {
            "language": language.value,
            "translations": translations,
            "available_languages": translator.get_available_languages()
        }
    except Exception as e:
        logger.error("Error getting translations", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics(
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener métricas del sistema
    """
    try:
        metrics = metrics_collector.get_metrics()
        return metrics
    except Exception as e:
        logger.error("Error getting metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/prometheus")
async def get_prometheus_metrics(
    user: User = Depends(current_user),
) -> Response:
    """
    Obtener métricas en formato Prometheus
    """
    try:
        prometheus_metrics = metrics_collector.get_prometheus_metrics()
        if not prometheus_metrics:
            raise HTTPException(status_code=503, detail="Prometheus metrics not available")
        
        return Response(
            content=prometheus_metrics,
            media_type="text/plain; version=0.0.4"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting Prometheus metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab/experiments")
async def create_ab_experiment(
    name: str = Query(..., description="Nombre del experimento"),
    description: str = Query(..., description="Descripción"),
    variants: List[Variant] = Query(..., description="Variantes"),
    traffic_split: Dict[Variant, float] = Query(..., description="División de tráfico"),
    user: User = Depends(current_user),
) -> dict:
    """
    Crear experimento A/B
    """
    try:
        from datetime import datetime
        experiment = ab_test_manager.create_experiment(
            name=name,
            description=description,
            variants=variants,
            traffic_split=traffic_split,
            start_date=datetime.utcnow()
        )
        
        return {
            "experiment_id": str(experiment.id),
            "name": experiment.name,
            "status": experiment.status.value
        }
    except Exception as e:
        logger.error("Error creating AB experiment", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab/experiments/{experiment_id}/assign")
async def assign_ab_variant(
    experiment_id: UUID = Path(..., description="ID del experimento"),
    user: User = Depends(current_user),
) -> dict:
    """
    Asignar variante A/B a usuario
    """
    try:
        variant = ab_test_manager.assign_variant(experiment_id, user.id)
        return {
            "experiment_id": str(experiment_id),
            "user_id": str(user.id),
            "variant": variant.value
        }
    except Exception as e:
        logger.error("Error assigning AB variant", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab/experiments/{experiment_id}/results")
async def get_ab_results(
    experiment_id: UUID = Path(..., description="ID del experimento"),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener resultados de experimento A/B
    """
    try:
        results = ab_test_manager.get_experiment_results(experiment_id)
        return results
    except Exception as e:
        logger.error("Error getting AB results", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/history")
async def get_event_history(
    event_type: Optional[EventType] = Query(None, description="Filtrar por tipo"),
    limit: int = Query(100, ge=1, le=1000),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener historial de eventos
    """
    try:
        events = event_bus.get_event_history(event_type=event_type, limit=limit)
        return {
            "events": [e.to_dict() for e in events],
            "total": len(events)
        }
    except Exception as e:
        logger.error("Error getting event history", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/versions")
async def get_api_versions(
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener información de versiones de API
    """
    try:
        versions = api_version_manager.get_all_versions()
        latest = api_version_manager.get_latest_version()
        
        return {
            "versions": [v.to_dict() for v in versions],
            "latest": latest.to_dict(),
            "current": APIVersion.LATEST.value
        }
    except Exception as e:
        logger.error("Error getting API versions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/migrations/status")
async def get_migration_status(
    version: Optional[str] = Query(None, description="Versión específica"),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener estado de migraciones
    """
    try:
        status = migration_manager.get_migration_status(version=version)
        return status
    except Exception as e:
        logger.error("Error getting migration status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/migrations/{version}/apply")
async def apply_migration(
    version: str = Path(..., description="Versión de migración"),
    user: User = Depends(current_user),
) -> dict:
    """
    Aplicar migración
    """
    try:
        success = await migration_manager.apply_migration(version)
        if not success:
            raise HTTPException(status_code=500, detail="Migration failed")
        
        return {
            "version": version,
            "status": "applied",
            "message": "Migration applied successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Error applying migration", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/validate")
async def validate_data(
    data: Dict[str, Any] = None,
    schema: Optional[Dict[str, str]] = None,
    user: User = Depends(current_user),
) -> dict:
    """
    Validar datos
    """
    try:
        result = data_validator.validate_data(data or {}, schema or {})
        return result
    except Exception as e:
        logger.error("Error validating data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/transform")
async def transform_data(
    data: Dict[str, Any] = None,
    transformations: Dict[str, str] = None,
    user: User = Depends(current_user),
) -> dict:
    """
    Transformar datos
    """
    try:
        if not data or not transformations:
            raise HTTPException(status_code=400, detail="Data and transformations required")
        
        result = data_transformer.transform_dict(data, transformations)
        return {"transformed_data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error transforming data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync")
async def start_sync(
    source: str = Query(..., description="Fuente de datos"),
    target: str = Query(..., description="Destino de datos"),
    data_type: str = Query(..., description="Tipo de dato"),
    sync_type: str = Query("full", description="Tipo de sincronización"),
    user: User = Depends(current_user),
) -> dict:
    """
    Iniciar sincronización
    """
    try:
        task_id = await sync_manager.sync(source, target, data_type, sync_type)
        return {
            "task_id": str(task_id),
            "status": "started",
            "message": "Sync task started"
        }
    except Exception as e:
        logger.error("Error starting sync", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sync/{task_id}")
async def get_sync_status(
    task_id: UUID = Path(..., description="ID de tarea"),
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener estado de sincronización
    """
    try:
        status = sync_manager.get_sync_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Sync task not found")
        
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting sync status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deep-learning/analyze")
async def analyze_with_deep_learning(
    texts: List[str] = Query(..., description="Textos a analizar"),
    include_llm: bool = Query(True, description="Incluir análisis con LLM"),
    user: User = Depends(current_user),
) -> dict:
    """
    Análisis completo usando modelos de deep learning
    """
    try:
        results = await deep_learning_analyzer.analyze_comprehensive(
            texts=texts,
            include_llm=include_llm
        )
        
        return {
            "analysis": results,
            "model_info": {
                "embedding_model": deep_learning_analyzer.embedding_model.model_name,
                "personality_model": deep_learning_analyzer.personality_classifier.model_name,
                "sentiment_model": deep_learning_analyzer.sentiment_model.model_name,
                "device": str(deep_learning_analyzer.device)
            }
        }
    except Exception as e:
        logger.error("Error in deep learning analysis", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fine-tuning/train")
async def train_model(
    texts: List[str] = Query(..., description="Textos de entrenamiento"),
    labels: Dict[str, List[float]] = Query(..., description="Etiquetas"),
    num_epochs: int = Query(3, ge=1, le=10),
    batch_size: int = Query(16, ge=1, le=64),
    learning_rate: float = Query(2e-5, ge=1e-6, le=1e-3),
    user: User = Depends(current_user),
) -> dict:
    """
    Entrenar modelo con fine-tuning
    """
    try:
        # Preparar dataset
        train_dataset = lora_trainer.prepare_dataset(texts, labels)
        
        # Entrenar
        results = lora_trainer.train(
            train_dataset=train_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        return {
            "training_results": results,
            "message": "Model training completed"
        }
    except Exception as e:
        logger.error("Error training model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visualization/generate")
async def generate_visualization(
    personality_traits: Dict[str, float] = None,
    sentiment_data: Optional[Dict[str, Any]] = None,
    visualization_type: str = Query("profile", description="Tipo: profile o sentiment"),
    user: User = Depends(current_user),
) -> Response:
    """
    Generar visualización usando diffusion models
    """
    try:
        if visualization_type == "profile" and personality_traits:
            image = visualization_generator.generate_profile_visualization(personality_traits)
        elif visualization_type == "sentiment" and sentiment_data:
            image = visualization_generator.generate_sentiment_visualization(sentiment_data)
        else:
            raise HTTPException(status_code=400, detail="Invalid visualization type or missing data")
        
        if image is None:
            raise HTTPException(status_code=503, detail="Visualization generation failed")
        
        # Convertir a bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        return Response(
            content=img_bytes.getvalue(),
            media_type="image/png"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating visualization", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gradio/launch")
async def launch_gradio_interface(
    user: User = Depends(current_user),
) -> dict:
    """
    Lanzar interfaz Gradio (información)
    """
    try:
        return {
            "message": "Gradio interface available",
            "endpoint": "/gradio",
            "instructions": "Launch Gradio interface separately using: gradio_interface.launch()"
        }
    except Exception as e:
        logger.error("Error with Gradio interface", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/track")
async def track_experiment(
    metrics: Dict[str, float] = None,
    step: Optional[int] = None,
    user: User = Depends(current_user),
) -> dict:
    """
    Registrar métricas de experimento
    """
    try:
        experiment_tracker.log_metrics(metrics or {}, step=step)
        return {
            "message": "Metrics logged",
            "wandb_enabled": experiment_tracker.use_wandb,
            "tensorboard_enabled": experiment_tracker.use_tensorboard
        }
    except Exception as e:
        logger.error("Error tracking experiment", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluation/evaluate")
async def evaluate_model(
    model_type: str = Query(..., description="Tipo: classification, regression, personality"),
    data_loader_config: Optional[Dict[str, Any]] = None,
    user: User = Depends(current_user),
) -> dict:
    """
    Evaluar modelo con métricas completas
    """
    try:
        # En producción, cargar modelo y data loader reales
        # Por ahora, retornar estructura de ejemplo
        return {
            "message": "Evaluation endpoint - implement with actual model and data",
            "model_type": model_type,
            "note": "This requires model and data loader to be properly configured"
        }
    except Exception as e:
        logger.error("Error evaluating model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/checkpoints/save")
async def save_checkpoint(
    epoch: int = Query(..., description="Epoch number"),
    train_loss: float = Query(..., description="Training loss"),
    val_loss: Optional[float] = Query(None, description="Validation loss"),
    user: User = Depends(current_user),
) -> dict:
    """
    Guardar checkpoint del modelo
    """
    try:
        # En producción, usar modelo y optimizer reales
        return {
            "message": "Checkpoint save endpoint - implement with actual model",
            "note": "This requires model and optimizer to be properly configured"
        }
    except Exception as e:
        logger.error("Error saving checkpoint", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/checkpoints/list")
async def list_checkpoints(
    user: User = Depends(current_user),
) -> dict:
    """
    Listar checkpoints disponibles
    """
    try:
        checkpoints = checkpoint_manager.checkpoint_history
        return {
            "checkpoints": checkpoints,
            "best_model": str(checkpoint_manager.checkpoint_dir / "best_model.pt") if (checkpoint_manager.checkpoint_dir / "best_model.pt").exists() else None
        }
    except Exception as e:
        logger.error("Error listing checkpoints", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/predict")
async def predict_with_inference_engine(
    texts: List[str] = Query(..., description="Textos para inferencia"),
    return_probs: bool = Query(False, description="Retornar probabilidades"),
    user: User = Depends(current_user),
) -> dict:
    """
    Predecir usando motor de inferencia optimizado
    """
    try:
        # En producción, usar inference engine real
        return {
            "message": "Inference endpoint - implement with actual inference engine",
            "texts_count": len(texts),
            "note": "This requires model and tokenizer to be properly configured"
        }
    except Exception as e:
        logger.error("Error in inference", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiling/stats")
async def get_profiling_stats(
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener estadísticas de profiling
    """
    try:
        memory_stats = performance_profiler.get_memory_stats()
        return {
            "profiling_data": performance_profiler.profiling_data,
            "memory_stats": memory_stats
        }
    except Exception as e:
        logger.error("Error getting profiling stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/debug/check-gradients")
async def check_gradients(
    check_nan: bool = Query(True, description="Check for NaN"),
    check_inf: bool = Query(True, description="Check for Inf"),
    check_exploding: bool = Query(True, description="Check for exploding gradients"),
    user: User = Depends(current_user),
) -> dict:
    """
    Verificar gradientes del modelo
    """
    try:
        # En producción, usar modelo real
        return {
            "message": "Gradient check endpoint - implement with actual model",
            "note": "This requires model to be properly configured"
        }
    except Exception as e:
        logger.error("Error checking gradients", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/augmentation/augment")
async def augment_texts(
    texts: List[str] = Query(..., description="Textos para aumentar"),
    methods: Optional[List[str]] = Query(None, description="Métodos de augmentación"),
    num_augmentations: int = Query(1, ge=1, le=10, description="Número de aumentaciones"),
    user: User = Depends(current_user),
) -> dict:
    """
    Aumentar textos para entrenamiento
    """
    try:
        augmented_results = []
        for text in texts:
            augmented = text_augmenter.augment(
                text,
                methods=methods,
                num_augmentations=num_augmentations
            )
            augmented_results.append({
                "original": text,
                "augmented": augmented
            })
        
        return {
            "augmented_texts": augmented_results,
            "total": len(augmented_results)
        }
    except Exception as e:
        logger.error("Error augmenting texts", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ensemble/predict")
async def ensemble_predict(
    texts: List[str] = Query(..., description="Textos para predicción"),
    voting_strategy: str = Query("average", description="Estrategia: average, weighted, majority"),
    user: User = Depends(current_user),
) -> dict:
    """
    Predecir usando ensemble de modelos
    """
    try:
        # En producción, usar ensemble real
        return {
            "message": "Ensemble prediction endpoint - implement with actual ensemble",
            "note": "This requires ensemble models to be properly configured"
        }
    except Exception as e:
        logger.error("Error in ensemble prediction", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transfer-learning/freeze")
async def freeze_model_layers(
    freeze_layers: Optional[List[str]] = Query(None, description="Capas a congelar"),
    freeze_all: bool = Query(False, description="Congelar todas las capas"),
    user: User = Depends(current_user),
) -> dict:
    """
    Congelar capas del modelo para transfer learning
    """
    try:
        # En producción, usar modelo real
        return {
            "message": "Freeze layers endpoint - implement with actual model",
            "freeze_all": freeze_all,
            "freeze_layers": freeze_layers
        }
    except Exception as e:
        logger.error("Error freezing layers", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hyperparameter-tuning/optimize")
async def optimize_hyperparameters(
    strategy: str = Query("random", description="Estrategia: grid, random, bayesian"),
    n_trials: int = Query(20, ge=1, le=100),
    user: User = Depends(current_user),
) -> dict:
    """
    Optimizar hiperparámetros
    """
    try:
        # En producción, implementar objective function real
        return {
            "message": "Hyperparameter tuning endpoint - implement with actual objective function",
            "strategy": strategy,
            "n_trials": n_trials
        }
    except Exception as e:
        logger.error("Error in hyperparameter tuning", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export/pytorch")
async def export_pytorch_model(
    model_name: str = Query(..., description="Nombre del modelo"),
    metadata: Optional[Dict[str, Any]] = None,
    user: User = Depends(current_user),
) -> dict:
    """
    Exportar modelo PyTorch
    """
    try:
        # En producción, usar modelo real
        return {
            "message": "Export endpoint - implement with actual model",
            "model_name": model_name
        }
    except Exception as e:
        logger.error("Error exporting model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export/onnx")
async def export_onnx_model(
    model_name: str = Query(..., description="Nombre del modelo"),
    input_shape: tuple = Query((1, 512), description="Shape de entrada"),
    user: User = Depends(current_user),
) -> dict:
    """
    Exportar modelo ONNX
    """
    try:
        # En producción, usar modelo real
        return {
            "message": "ONNX export endpoint - implement with actual model",
            "model_name": model_name,
            "input_shape": input_shape
        }
    except Exception as e:
        logger.error("Error exporting ONNX model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/stats")
async def get_memory_stats(
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener estadísticas de memoria
    """
    try:
        stats = memory_optimizer.get_memory_usage()
        return {
            "memory_stats": stats,
            "cache_cleared": False
        }
    except Exception as e:
        logger.error("Error getting memory stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/clear-cache")
async def clear_memory_cache(
    user: User = Depends(current_user),
) -> dict:
    """
    Limpiar caché de memoria
    """
    try:
        memory_optimizer.clear_cache()
        return {
            "message": "Memory cache cleared",
            "cache_cleared": True
        }
    except Exception as e:
        logger.error("Error clearing cache", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/diffusion/advanced/generate")
async def generate_with_advanced_diffusion(
    prompt: str = Query(..., description="Prompt para generación"),
    negative_prompt: Optional[str] = Query(None, description="Negative prompt"),
    scheduler_type: str = Query("dpm", description="Tipo de scheduler: dpm, ddim, euler, pndm"),
    num_inference_steps: int = Query(50, ge=10, le=100),
    guidance_scale: float = Query(7.5, ge=1.0, le=20.0),
    seed: Optional[int] = Query(None, description="Seed para reproducibilidad"),
    user: User = Depends(current_user),
) -> Response:
    """
    Generar imagen con pipeline de difusión avanzado
    """
    try:
        # Configurar scheduler
        advanced_diffusion_pipeline.scheduler_type = scheduler_type
        advanced_diffusion_pipeline._setup_scheduler()
        
        image = advanced_diffusion_pipeline.generate_with_control(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        if image is None:
            raise HTTPException(status_code=503, detail="Image generation failed")
        
        # Convertir a bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        return Response(
            content=img_bytes.getvalue(),
            media_type="image/png"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating image with advanced diffusion", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validation/validate-model")
async def validate_model_comprehensive(
    metrics: Optional[List[str]] = Query(None, description="Métricas a calcular"),
    user: User = Depends(current_user),
) -> dict:
    """
    Validar modelo de forma completa
    """
    try:
        # En producción, usar modelo y data loader reales
        return {
            "message": "Model validation endpoint - implement with actual model and data",
            "metrics": metrics or ["accuracy", "precision", "recall", "f1"]
        }
    except Exception as e:
        logger.error("Error validating model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validation/validate-gradients")
async def validate_gradients_comprehensive(
    max_norm: float = Query(10.0, ge=0.1, le=100.0),
    user: User = Depends(current_user),
) -> dict:
    """
    Validar gradientes del modelo
    """
    try:
        # En producción, usar modelo real
        return {
            "message": "Gradient validation endpoint - implement with actual model",
            "max_norm": max_norm
        }
    except Exception as e:
        logger.error("Error validating gradients", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/create")
async def create_experiment(
    name: str = Query(..., description="Nombre del experimento"),
    description: str = Query("", description="Descripción"),
    tags: Optional[List[str]] = Query(None, description="Tags"),
    user: User = Depends(current_user),
) -> dict:
    """
    Crear nuevo experimento
    """
    try:
        experiment = experiment_manager.create_experiment(
            name=name,
            description=description,
            tags=tags
        )
        return {
            "experiment": experiment.to_dict(),
            "message": "Experiment created successfully"
        }
    except Exception as e:
        logger.error("Error creating experiment", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
async def list_experiments(
    tags: Optional[List[str]] = Query(None, description="Filtrar por tags"),
    status: Optional[str] = Query(None, description="Filtrar por status"),
    user: User = Depends(current_user),
) -> dict:
    """
    Listar experimentos
    """
    try:
        experiments = experiment_manager.list_experiments(tags=tags, status=status)
        return {
            "experiments": [e.to_dict() for e in experiments],
            "total": len(experiments)
        }
    except Exception as e:
        logger.error("Error listing experiments", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/registry")
async def list_registered_models(
    user: User = Depends(current_user),
) -> dict:
    """
    Listar modelos registrados
    """
    try:
        models = model_registry.list_models()
        return {
            "models": models,
            "total": len(models)
        }
    except Exception as e:
        logger.error("Error listing models", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/system")
async def get_system_monitoring(
    user: User = Depends(current_user),
) -> dict:
    """
    Obtener estadísticas del sistema
    """
    try:
        stats = system_monitor.get_system_stats()
        return {
            "system_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Error getting system stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(
    user: Optional[User] = None,  # Health check should be accessible
) -> dict:
    """
    Health check endpoint
    """
    try:
        health_status = health_checker.check_health()
        status_code = 200 if health_status["status"] == "healthy" else 503
        return Response(
            content=json.dumps(health_status),
            status_code=status_code,
            media_type="application/json"
        )
    except Exception as e:
        logger.error("Error in health check", error=str(e))
        return Response(
            content=json.dumps({"status": "error", "message": str(e)}),
            status_code=503,
            media_type="application/json"
        )


@router.post("/optimization/quantize")
async def quantize_model(
    quantization_type: str = Query("dynamic", description="Tipo: dynamic o static"),
    user: User = Depends(current_user),
) -> dict:
    """
    Cuantizar modelo para optimización
    """
    try:
        # En producción, usar modelo real
        return {
            "message": "Quantization endpoint - implement with actual model",
            "quantization_type": quantization_type,
            "note": "This requires model to be properly configured"
        }
    except Exception as e:
        logger.error("Error quantizing model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimization/prune")
async def prune_model(
    pruning_type: str = Query("unstructured", description="Tipo: structured o unstructured"),
    pruning_config: Optional[Dict[str, float]] = None,
    user: User = Depends(current_user),
) -> dict:
    """
    Podar modelo para reducir tamaño
    """
    try:
        # En producción, usar modelo real
        return {
            "message": "Pruning endpoint - implement with actual model",
            "pruning_type": pruning_type,
            "pruning_config": pruning_config or {},
            "note": "This requires model to be properly configured"
        }
    except Exception as e:
        logger.error("Error pruning model", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deployment/create")
async def create_deployment(
    model_name: str = Query(..., description="Nombre del modelo"),
    version: str = Query(..., description="Versión"),
    metadata: Optional[Dict[str, Any]] = None,
    user: User = Depends(current_user),
) -> dict:
    """
    Crear paquete de deployment
    """
    try:
        # En producción, usar modelo real
        return {
            "message": "Deployment creation endpoint - implement with actual model",
            "model_name": model_name,
            "version": version,
            "note": "This requires model to be properly configured"
        }
    except Exception as e:
        logger.error("Error creating deployment", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployment/{model_name}/versions")
async def list_model_versions(
    model_name: str = Path(..., description="Nombre del modelo"),
    user: User = Depends(current_user),
) -> dict:
    """
    Listar versiones de un modelo
    """
    try:
        versions = model_versioning.list_versions(model_name)
        return {
            "model_name": model_name,
            "versions": versions,
            "total": len(versions)
        }
    except Exception as e:
        logger.error("Error listing versions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark/inference")
async def benchmark_inference(
    num_runs: int = Query(100, ge=10, le=1000),
    warmup_runs: int = Query(10, ge=1, le=100),
    user: User = Depends(current_user),
) -> dict:
    """
    Benchmark de inferencia
    """
    try:
        # En producción, usar modelo y datos reales
        return {
            "message": "Inference benchmark endpoint - implement with actual model and data",
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
            "note": "This requires model and input data to be properly configured"
        }
    except Exception as e:
        logger.error("Error in inference benchmark", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security/compute-hash")
async def compute_model_hash(
    algorithm: str = Query("sha256", description="Algoritmo: sha256 o md5"),
    user: User = Depends(current_user),
) -> dict:
    """
    Calcular hash del modelo
    """
    try:
        # En producción, usar modelo real
        return {
            "message": "Hash computation endpoint - implement with actual model",
            "algorithm": algorithm,
            "note": "This requires model to be properly configured"
        }
    except Exception as e:
        logger.error("Error computing hash", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security/verify-integrity")
async def verify_model_integrity(
    expected_hash: str = Query(..., description="Hash esperado"),
    algorithm: str = Query("sha256", description="Algoritmo"),
    user: User = Depends(current_user),
) -> dict:
    """
    Verificar integridad del modelo
    """
    try:
        # En producción, usar modelo real
        return {
            "message": "Integrity verification endpoint - implement with actual model",
            "expected_hash": expected_hash,
            "algorithm": algorithm,
            "note": "This requires model to be properly configured"
        }
    except Exception as e:
        logger.error("Error verifying integrity", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Nota: El middleware de rate limiting debe agregarse a la aplicación FastAPI principal
# No se puede agregar directamente al router

