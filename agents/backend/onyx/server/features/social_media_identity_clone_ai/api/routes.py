"""
Rutas de API para Social Media Identity Clone AI
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
import logging
# Removed: hashlib, OrderedDict - now using helpers
from functools import lru_cache

from ..services.profile_extractor import ProfileExtractor
from ..services.identity_analyzer import IdentityAnalyzer
from ..services.content_generator import ContentGenerator
from ..services.storage_service import StorageService
from ..core.models import IdentityProfile, Platform, ContentType
from ..db.base import init_db, get_db_session
from ..analytics.metrics import get_metrics_collector
from ..analytics.analytics_service import AnalyticsService
from ..services.webhook_service import get_webhook_service
from ..services.export_service import ExportService
from ..services.versioning_service import VersioningService
from ..services.batch_service import BatchService
from ..queue.task_queue import get_task_queue, TaskStatus
from ..search.search_service import SearchService, SearchFilter
from ..templates.template_service import get_template_service, ContentTemplate
from ..notifications.notification_service import get_notification_service, NotificationType
from ..recommendations.recommendation_service import get_recommendation_service
from ..scheduler.scheduler_service import get_scheduler_service, ScheduleType
from ..ab_testing.ab_test_service import get_ab_test_service
from ..backup.backup_service import get_backup_service
from ..services.content_validator import ContentValidator
from ..ml.ml_service import get_ml_service
from ..collaboration.collaboration_service import get_collaboration_service, PermissionLevel
from ..dashboard.dashboard_service import get_dashboard_service
from ..alerts.alert_service import get_alert_service, AlertType, AlertSeverity
from ..plugins.plugin_service import get_plugin_service
from ..utils.performance_monitor import get_performance_monitor
from ..utils.health_check import get_health_check
from .decorators import handle_api_errors, log_endpoint_call, cache_response

# Helper imports
from ..utils.cache_helpers import generate_cache_key
from ..utils.cache_manager import get_cache
from ..utils.serialization_helpers import serialize_model
from ..utils.metrics_helpers import track_operation
from ..utils.platform_helpers import execute_for_platform
from ..utils.validation_helpers import validate_platform, validate_content_type
from ..utils.webhook_helpers import send_webhook
from .response_helpers import success_response
from .exception_helpers import not_found, validation_error

logger = logging.getLogger(__name__)

router = APIRouter()

# Response cache para endpoints GET - Using helper
cache = get_cache()

# Lazy loading de servicios - Using helper
from ..core.service_factory import create_service_getter

get_analytics_service = create_service_getter(AnalyticsService)
get_export_service = create_service_getter(ExportService)
get_versioning_service = create_service_getter(VersioningService)
get_batch_service = create_service_getter(BatchService)
get_search_service = create_service_getter(SearchService)

# Servicios que se inicializan al importar (necesarios para startup)
metrics = get_metrics_collector()
performance_monitor = get_performance_monitor()
health_check = get_health_check()

# Inicializar base de datos al importar
init_db()

# Iniciar monitor de rendimiento
performance_monitor.start_monitoring()


class ExtractProfileRequest(BaseModel):
    """Request para extraer perfil"""
    platform: str = Field(..., description="Plataforma: tiktok, instagram, youtube")
    username: str = Field(..., min_length=1, max_length=255)
    use_cache: bool = Field(True, description="Usar caché si está disponible")
    
    @validator("platform")
    def validate_platform(cls, v):
        valid_platforms = ["tiktok", "instagram", "youtube"]
        if v.lower() not in valid_platforms:
            raise ValueError(f"Plataforma debe ser una de: {', '.join(valid_platforms)}")
        return v.lower()


class BuildIdentityRequest(BaseModel):
    """Request para construir identidad"""
    tiktok_username: Optional[str] = None
    instagram_username: Optional[str] = None
    youtube_channel_id: Optional[str] = None


class GenerateContentRequest(BaseModel):
    """Request para generar contenido"""
    identity_profile_id: str
    platform: str
    content_type: str
    topic: Optional[str] = None
    style: Optional[str] = None
    duration: Optional[int] = None
    video_title: Optional[str] = None
    tags: Optional[List[str]] = None


@router.post("/extract-profile", status_code=status.HTTP_200_OK)
@handle_api_errors
@log_endpoint_call
async def extract_profile(request: ExtractProfileRequest):
    """
    Extrae perfil de una red social
    Optimizado con caché, lazy loading y decoradores
    
    Args:
        request: Datos del perfil a extraer
        
    Returns:
        Perfil extraído
    """
    logger.info(f"Extrayendo perfil: {request.platform}/{request.username}")
    
    # Cache check using helper
    cache_key = generate_cache_key("extract_profile", request.platform, request.username)
    if request.use_cache:
        cached_response = cache.get(cache_key)
        if cached_response:
            return cached_response
    
    # Metrics tracking using helper
    with track_operation("profile_extraction", tags={"platform": request.platform}):
        extractor = ProfileExtractor()
        
        # Platform handler mapping using helper
        platform_map = {
            "tiktok": extractor.extract_tiktok_profile,
            "instagram": extractor.extract_instagram_profile,
            "youtube": extractor.extract_youtube_profile
        }
        
        profile = await execute_for_platform(
            request.platform,
            platform_map,
            request.username,
            use_cache=request.use_cache
        )
        
        if not profile:
            raise validation_error(
                f"Plataforma no soportada: {request.platform}",
                field="platform"
            )
        
        # Response formatting using helpers
        response = success_response(
            data={
                "platform": request.platform,
                "username": request.username,
                "profile": serialize_model(profile)
            },
            metadata={
                "stats": {
                    "videos": len(profile.videos),
                    "posts": len(profile.posts),
                    "comments": len(profile.comments)
                }
            }
        )
        
        # Store in cache
        if request.use_cache:
            cache.set(cache_key, response)
        
        return response


@router.post("/build-identity", status_code=status.HTTP_201_CREATED)
@handle_api_errors
@log_endpoint_call
async def build_identity(request: BuildIdentityRequest):
    """
    Construye perfil de identidad a partir de perfiles de redes sociales
    Refactorizado con decoradores para manejo de errores
    
    Args:
        request: Datos de perfiles a usar
        
    Returns:
        Perfil de identidad construido
    """
    logger.info("Construyendo identidad...")
    
    # Metrics tracking using helper
    with track_operation("identity_build"):
        extractor = ProfileExtractor()
        analyzer = IdentityAnalyzer()
        storage = StorageService()
    
    # Validation using helper
    from ..utils.validation_helpers import validate_at_least_one
    validate_at_least_one(
        request.tiktok_username,
        request.instagram_username,
        request.youtube_channel_id,
        field_names=["tiktok_username", "instagram_username", "youtube_channel_id"],
        message="Al menos un perfil debe ser proporcionado"
    )
    
    # Extraer perfiles
    tiktok_profile = None
    instagram_profile = None
    youtube_profile = None
    
    if request.tiktok_username:
        logger.info(f"Extrayendo perfil de TikTok: {request.tiktok_username}")
        tiktok_profile = await extractor.extract_tiktok_profile(request.tiktok_username)
    
    if request.instagram_username:
        logger.info(f"Extrayendo perfil de Instagram: {request.instagram_username}")
        instagram_profile = await extractor.extract_instagram_profile(request.instagram_username)
    
    if request.youtube_channel_id:
        logger.info(f"Extrayendo canal de YouTube: {request.youtube_channel_id}")
        youtube_profile = await extractor.extract_youtube_profile(request.youtube_channel_id)
    
    # Construir identidad
    logger.info("Analizando contenido y construyendo identidad...")
    identity = await analyzer.build_identity(
        tiktok_profile=tiktok_profile,
        instagram_profile=instagram_profile,
        youtube_profile=youtube_profile
    )
    
    # Guardar en base de datos
    storage.save_identity(identity)
    
    # Send webhook using helper
    await send_webhook("identity_created", {
        "identity_id": identity.profile_id,
        "username": identity.username,
        "stats": {
            "total_videos": identity.total_videos,
            "total_posts": identity.total_posts,
            "total_comments": identity.total_comments
        }
    })
    
    # Response formatting using helpers
    return success_response(
        data={
            "identity_id": identity.profile_id,
            "identity": serialize_model(identity)
        },
        metadata={
            "stats": {
                "total_videos": identity.total_videos,
                "total_posts": identity.total_posts,
                "total_comments": identity.total_comments,
                "topics_count": len(identity.content_analysis.topics),
                "themes_count": len(identity.content_analysis.themes)
            }
        }
    )


@router.post("/generate-content", status_code=status.HTTP_201_CREATED)
async def generate_content(request: GenerateContentRequest):
    """
    Genera contenido basado en identidad clonada
    
    Args:
        request: Datos para generación de contenido
        
    Returns:
        Contenido generado
    """
    logger.info(f"Generando contenido para identidad: {request.identity_profile_id}")
    
    # Metrics tracking using helper
    with track_operation(
        "content_generation",
        tags={
            "platform": request.platform,
            "content_type": request.content_type
        }
    ):
        storage = StorageService()
        
        # Load identity using helper for error handling
        identity = storage.get_identity(request.identity_profile_id)
        if not identity:
            raise not_found("Identidad", request.identity_profile_id)
        
        # Validate platform and content type using helpers
        platform = validate_platform(request.platform)
        content_type = validate_content_type(request.content_type)
        
        # Generar contenido usando platform/content type mapping
        generator = ContentGenerator(identity_profile=identity)
        
        # Map platform + content_type to generator methods
        content_handlers = {
            (Platform.INSTAGRAM, ContentType.POST): lambda: generator.generate_instagram_post(
                topic=request.topic,
                style=request.style
            ),
            (Platform.TIKTOK, ContentType.VIDEO): lambda: generator.generate_tiktok_script(
                topic=request.topic,
                duration=request.duration or 60
            ),
            (Platform.YOUTUBE, ContentType.VIDEO): lambda: generator.generate_youtube_description(
                video_title=request.video_title or "Video",
                tags=request.tags
            )
        }
        
        handler = content_handlers.get((platform, content_type))
        if not handler:
            raise validation_error(
                f"Combinación de plataforma y tipo no soportada: {platform.value}/{content_type.value}",
                field="platform,content_type"
            )
        
        generated = await handler()
        
        # Validar contenido antes de guardar
        validation = content_validator.validate(generated)
        
        # Agregar score de validación al metadata usando helper
        from ..utils.condition_helpers import if_none
        
        generated.metadata = if_none(generated.metadata, {})
        generated.metadata["validation"] = {
            "is_valid": validation.is_valid,
            "score": validation.score,
            "issues": validation.issues,
            "warnings": validation.warnings,
            "suggestions": validation.suggestions
        }
        
        # Guardar contenido generado
        storage.save_generated_content(generated)
        
        # Send webhook using helper
        await send_webhook("content_generated", {
            "content_id": generated.content_id,
            "identity_id": request.identity_profile_id,
            "platform": request.platform,
            "content_type": request.content_type,
            "validation_score": validation.score
        })
        
        # Response formatting using helpers
        return success_response(
            data={
                "content_id": generated.content_id,
                "content": serialize_model(generated)
            },
            metadata={
                "validation": {
                    "is_valid": validation.is_valid,
                    "score": validation.score,
                    "issues": validation.issues,
                    "warnings": validation.warnings,
                    "suggestions": validation.suggestions
                }
            }
        )


@router.get("/identity/{identity_id}", status_code=status.HTTP_200_OK)
@handle_api_errors
@log_endpoint_call
async def get_identity(identity_id: str):
    """
    Obtiene perfil de identidad por ID
    Optimizado con caché de respuesta y decoradores
    
    Args:
        identity_id: ID del perfil de identidad
        
    Returns:
        Perfil de identidad
    """
    # Verificar caché usando helper
    cache_key = generate_cache_key("get_identity", identity_id)
    cached_response = cache.get(cache_key)
    if cached_response:
        return cached_response
    
    storage = StorageService()
    identity = storage.get_identity(identity_id)
    
    if not identity:
        raise not_found("Identidad", identity_id)
    
    # Response formatting using helpers
    response = success_response(
        data={"identity": serialize_model(identity)}
    )
    
    # Guardar en caché
    cache.set(cache_key, response)
    
    return response


@router.get("/identity/{identity_id}/generated-content", status_code=status.HTTP_200_OK)
async def get_generated_content(identity_id: str, limit: int = 10):
    """
    Obtiene contenido generado para una identidad
    
    Args:
        identity_id: ID de la identidad
        limit: Límite de resultados (default: 10)
        
    Returns:
        Lista de contenido generado
    """
    from ..utils.error_handling_helpers import handle_errors
    from ..utils.serialization_helpers import serialize_models
    from .exception_helpers import internal_error
    
    storage = StorageService()
    
    # Verificar que la identidad existe
    identity = storage.get_identity(identity_id)
    if not identity:
        raise not_found("Identidad", identity_id)
    
    content_list = storage.get_generated_content(identity_id, limit=limit)
    
    # Response formatting using helpers
    return success_response(
        data={
            "identity_id": identity_id,
            "content": serialize_models(content_list)
        },
        metadata={"count": len(content_list)}
    )

