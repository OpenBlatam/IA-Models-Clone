"""
Servicios de negocio para Validación Psicológica AI
====================================================
"""

from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
import structlog
import asyncio
from enum import Enum

from .models import (
    PsychologicalValidation,
    SocialMediaConnection,
    SocialMediaPlatform,
    ValidationReport,
    PsychologicalProfile,
    ConnectionStatus,
    ValidationStatus,
)
from .schemas import (
    ValidationCreate,
    SocialMediaConnectRequest,
    SocialMediaDataRequest,
)
from .analyzers import AdvancedPsychologicalAnalyzer
from .social_media_clients import SocialMediaClientFactory
from .exceptions import (
    ValidationNotFoundError,
    ValidationAlreadyRunningError,
    InsufficientDataError,
    AnalysisTimeoutError,
    ProfileGenerationError,
    ReportGenerationError,
    SocialMediaConnectionError,
)
from .config import config
from .distributed_cache import distributed_cache
from .event_bus import event_bus, EventType
from .metrics import metrics_collector
from .deep_learning_models import deep_learning_analyzer

logger = structlog.get_logger()


class PsychologicalValidationService:
    """Servicio principal para validación psicológica"""

    def __init__(self):
        """Inicializar servicio"""
        self._connections: Dict[UUID, Dict[SocialMediaPlatform, SocialMediaConnection]] = {}
        self._validations: Dict[UUID, PsychologicalValidation] = {}
        self._analyzer = AdvancedPsychologicalAnalyzer()
        self._cache: Dict[str, Any] = {}
        logger.info("PsychologicalValidationService initialized", config=config.app_name)

    async def connect_social_media(
        self,
        user_id: UUID,
        request: SocialMediaConnectRequest
    ) -> SocialMediaConnection:
        """
        Conectar una red social del usuario
        
        Args:
            user_id: ID del usuario
            request: Datos de conexión
            
        Returns:
            Conexión creada
        """
        logger.info(
            "Connecting social media",
            user_id=str(user_id),
            platform=request.platform.value
        )

        # Crear conexión
        connection = SocialMediaConnection(
            user_id=user_id,
            platform=request.platform,
            access_token=request.access_token,
            refresh_token=request.refresh_token,
            status=ConnectionStatus.CONNECTED,
            connected_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=request.expires_in or 3600)
            if request.expires_in else None
        )

        # Obtener datos del perfil
        try:
            profile_data = await self._fetch_profile_data(connection)
            connection.profile_data = profile_data
            connection.last_sync_at = datetime.utcnow()
        except Exception as e:
            logger.error(
                "Error fetching profile data",
                error=str(e),
                platform=request.platform.value
            )
            connection.status = ConnectionStatus.ERROR

        # Almacenar conexión (en producción, usar base de datos)
        if user_id not in self._connections:
            self._connections[user_id] = {}
        self._connections[user_id][request.platform] = connection

        logger.info(
            "Social media connected",
            connection_id=str(connection.id),
            platform=request.platform.value
        )

        return connection

    async def disconnect_social_media(
        self,
        user_id: UUID,
        platform: SocialMediaPlatform
    ) -> bool:
        """
        Desconectar una red social
        
        Args:
            user_id: ID del usuario
            platform: Plataforma a desconectar
            
        Returns:
            True si se desconectó exitosamente
        """
        logger.info("Disconnecting social media", user_id=str(user_id), platform=platform.value)

        if user_id in self._connections:
            if platform in self._connections[user_id]:
                connection = self._connections[user_id][platform]
                connection.update_status(ConnectionStatus.DISCONNECTED)
                del self._connections[user_id][platform]
                logger.info("Social media disconnected", platform=platform.value)
                return True

        return False

    async def get_user_connections(
        self,
        user_id: UUID
    ) -> List[SocialMediaConnection]:
        """
        Obtener todas las conexiones del usuario
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Lista de conexiones
        """
        if user_id not in self._connections:
            return []

        connections = list(self._connections[user_id].values())
        logger.info("Retrieved user connections", user_id=str(user_id), count=len(connections))
        return connections

    async def create_validation(
        self,
        user_id: UUID,
        request: ValidationCreate
    ) -> PsychologicalValidation:
        """
        Crear una nueva validación psicológica
        
        Args:
            user_id: ID del usuario
            request: Datos de la validación
            
        Returns:
            Validación creada
        """
        logger.info("Creating validation", user_id=str(user_id))
        
        # Verificar cuota
        try:
            from .quotas import quota_manager, QuotaType
            from .exceptions import PsychologicalValidationError
            allowed, quota_info = quota_manager.check_quota(user_id, QuotaType.VALIDATIONS_PER_DAY)
            if not allowed:
                raise PsychologicalValidationError(
                    f"Quota exceeded: {QuotaType.VALIDATIONS_PER_DAY.value}",
                    error_code="QUOTA_EXCEEDED",
                    details=quota_info
                )
        except PsychologicalValidationError:
            raise
        except Exception as e:
            logger.warning("Error checking quota", error=str(e))

        # Verificar conexiones disponibles
        available_connections = await self.get_user_connections(user_id)
        available_platforms = {conn.platform for conn in available_connections if conn.status == ConnectionStatus.CONNECTED}

        # Filtrar plataformas solicitadas que estén conectadas
        platforms_to_analyze = [
            p for p in request.platforms
            if p in available_platforms
        ] or list(available_platforms)

        # Crear validación
        validation = PsychologicalValidation(
            user_id=user_id,
            status=ValidationStatus.PENDING,
            connected_platforms=platforms_to_analyze,
            metadata={
                "include_historical_data": request.include_historical_data,
                "analysis_depth": request.analysis_depth
            }
        )

        # Almacenar validación (en producción, usar base de datos)
        self._validations[validation.id] = validation
        
        # Registrar uso de cuota
        try:
            from .quotas import quota_manager, QuotaType
            quota_manager.record_usage(user_id, QuotaType.VALIDATIONS_PER_DAY)
        except Exception as e:
            logger.warning("Error recording quota usage", error=str(e))
        
        # Publicar evento
        try:
            await event_bus.publish_event(
                EventType.VALIDATION_CREATED,
                {
                    "validation_id": str(validation.id),
                    "user_id": str(user_id),
                    "platforms": validation.connected_platforms
                }
            )
        except Exception as e:
            logger.warning("Error publishing event", error=str(e))
        
        # Registrar métrica
        try:
            metrics_collector.increment_counter("validations_created")
        except Exception as e:
            logger.warning("Error recording metric", error=str(e))
        
        # Crear versión inicial
        try:
            from .versioning import version_manager
            version_manager.create_version(
                validation,
                changes=["Initial validation created"]
            )
        except Exception as e:
            logger.warning("Error creating version", error=str(e))

        logger.info(
            "Validation created",
            validation_id=str(validation.id),
            platforms=len(platforms_to_analyze)
        )

        return validation

    async def run_validation(
        self,
        validation_id: UUID
    ) -> PsychologicalValidation:
        """
        Ejecutar análisis de validación psicológica
        
        Args:
            validation_id: ID de la validación
            
        Returns:
            Validación completada con perfil y reporte
        """
        logger.info("Running validation", validation_id=str(validation_id))

        if validation_id not in self._validations:
            raise ValidationNotFoundError(
                f"Validation {validation_id} not found",
                error_code="VALIDATION_NOT_FOUND",
                details={"validation_id": str(validation_id)}
            )

        validation = self._validations[validation_id]
        
        if validation.status == ValidationStatus.IN_PROGRESS:
            raise ValidationAlreadyRunningError(
                f"Validation {validation_id} is already running",
                error_code="VALIDATION_RUNNING",
                details={"validation_id": str(validation_id)}
            )
        
        validation.update_status(ValidationStatus.IN_PROGRESS)

        try:
            # Obtener datos de todas las plataformas conectadas
            social_media_data = await self._collect_social_media_data(
                validation.user_id,
                validation.connected_platforms
            )

            # Generar perfil psicológico
            profile = await self._generate_psychological_profile(
                validation.user_id,
                social_media_data
            )
            validation.profile = profile
            
            # Notificar a plugins sobre perfil generado
            try:
                from .plugins import plugin_manager
                await plugin_manager.notify_profile_generated(profile)
            except Exception as e:
                logger.warning("Error notifying plugins", error=str(e))

            # Generar reporte
            report = await self._generate_validation_report(
                validation.id,
                social_media_data,
                profile
            )
            validation.report = report

            validation.update_status(ValidationStatus.COMPLETED)
            
            # Crear nueva versión con cambios
            try:
                from .versioning import version_manager
                version_manager.create_version(
                    validation,
                    changes=["Validation completed", "Profile and report generated"]
                )
            except Exception as e:
                logger.warning("Error creating version", error=str(e))
            
            # Disparar webhooks
            try:
                from .webhooks import webhook_manager, WebhookEvent
                await webhook_manager.trigger_webhook(
                    WebhookEvent.VALIDATION_COMPLETED,
                    {
                        "validation_id": str(validation.id),
                        "user_id": str(validation.user_id),
                        "confidence_score": profile.confidence_score,
                        "platforms": [p.value for p in validation.connected_platforms]
                    }
                )
            except Exception as e:
                logger.warning("Error triggering webhook", error=str(e))
            
            # Disparar webhooks
            try:
                from .webhooks import webhook_manager, WebhookEvent
                await webhook_manager.trigger_webhook(
                    WebhookEvent.VALIDATION_COMPLETED,
                    {
                        "validation_id": str(validation.id),
                        "user_id": str(validation.user_id),
                        "confidence_score": profile.confidence_score,
                        "platforms": [p.value for p in validation.connected_platforms]
                    }
                )
            except Exception as e:
                logger.warning("Error triggering webhook", error=str(e))

            logger.info(
                "Validation completed",
                validation_id=str(validation_id),
                confidence_score=profile.confidence_score
            )

        except Exception as e:
            logger.error("Validation failed", validation_id=str(validation_id), error=str(e))
            validation.update_status(ValidationStatus.FAILED)
            raise

        return validation

    async def get_validation(
        self,
        validation_id: UUID
    ) -> Optional[PsychologicalValidation]:
        """
        Obtener una validación por ID
        
        Args:
            validation_id: ID de la validación
            
        Returns:
            Validación o None si no existe
        """
        # Intentar obtener del caché
        cache_key = distributed_cache._generate_key("validation", str(validation_id))
        cached = await distributed_cache.get(cache_key)
        
        if cached:
            logger.debug("Validation retrieved from cache", validation_id=str(validation_id))
            return PsychologicalValidation.model_validate(cached)
        
        # Obtener de almacenamiento
        validation = self._validations.get(validation_id)
        
        # Guardar en caché
        if validation:
            await distributed_cache.set(
                cache_key,
                validation.to_dict(),
                ttl=3600
            )
        
        return validation

    async def get_user_validations(
        self,
        user_id: UUID
    ) -> List[PsychologicalValidation]:
        """
        Obtener todas las validaciones de un usuario
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Lista de validaciones
        """
        validations = [
            v for v in self._validations.values()
            if v.user_id == user_id
        ]
        logger.info("Retrieved user validations", user_id=str(user_id), count=len(validations))
        return validations

    async def _fetch_profile_data(
        self,
        connection: SocialMediaConnection
    ) -> Dict[str, Any]:
        """
        Obtener datos del perfil de la red social
        
        Args:
            connection: Conexión a la red social
            
        Returns:
            Datos del perfil
        """
        logger.info("Fetching profile data", platform=connection.platform.value)

        try:
            # Usar cliente de red social
            client = SocialMediaClientFactory.create_client(connection)
            profile_data = await client.get_profile()
            
            # Actualizar última sincronización
            connection.last_sync_at = datetime.utcnow()
            
            return profile_data
        except Exception as e:
            logger.error(
                "Error fetching profile data",
                error=str(e),
                platform=connection.platform.value
            )
            raise SocialMediaConnectionError(
                f"Failed to fetch profile data: {str(e)}",
                error_code="PROFILE_FETCH_ERROR",
                details={"platform": connection.platform.value}
            )

    async def _collect_social_media_data(
        self,
        user_id: UUID,
        platforms: List[SocialMediaPlatform]
    ) -> Dict[SocialMediaPlatform, Dict[str, Any]]:
        """
        Recolectar datos de todas las plataformas conectadas
        
        Args:
            user_id: ID del usuario
            platforms: Lista de plataformas
            
        Returns:
            Diccionario con datos por plataforma
        """
        logger.info("Collecting social media data", platforms=[p.value for p in platforms])

        data = {}
        connections = await self.get_user_connections(user_id)

        for platform in platforms:
            connection = next(
                (c for c in connections if c.platform == platform and c.status == ConnectionStatus.CONNECTED),
                None
            )

            if connection:
                try:
                    platform_data = await self._fetch_platform_data(connection)
                    data[platform] = platform_data
                except Exception as e:
                    logger.error(
                        "Error collecting data from platform",
                        platform=platform.value,
                        error=str(e)
                    )

        return data

    async def _fetch_platform_data(
        self,
        connection: SocialMediaConnection
    ) -> Dict[str, Any]:
        """
        Obtener datos de una plataforma específica
        
        Args:
            connection: Conexión a la plataforma
            
        Returns:
            Datos de la plataforma
        """
        logger.info("Fetching platform data", platform=connection.platform.value)

        try:
            # Usar cliente de red social
            client = SocialMediaClientFactory.create_client(connection)
            
            # Obtener posts
            posts = await client.get_posts(limit=100)
            
            # Calcular interacciones totales
            total_likes = sum(p.get("likes", 0) for p in posts)
            total_comments = sum(p.get("comments", 0) for p in posts)
            total_shares = sum(p.get("shares", 0) for p in posts)
            
            # Analizar patrón de actividad
            if posts:
                post_times = [
                    datetime.fromisoformat(p.get("created_at", datetime.utcnow().isoformat()))
                    for p in posts
                ]
                hours = [pt.hour for pt in post_times]
                most_active_hour = max(set(hours), key=hours.count) if hours else 14
            else:
                most_active_hour = 14
            
            return {
                "posts": posts,
                "interactions": {
                    "total_likes": total_likes,
                    "total_comments": total_comments,
                    "total_shares": total_shares
                },
                "activity_pattern": {
                    "most_active_hour": most_active_hour,
                    "most_active_day": "Monday"  # Simplificado
                }
            }
        except Exception as e:
            logger.error(
                "Error fetching platform data",
                error=str(e),
                platform=connection.platform.value
            )
            # Retornar datos vacíos en caso de error
            return {
                "posts": [],
                "interactions": {
                    "total_likes": 0,
                    "total_comments": 0,
                    "total_shares": 0
                },
                "activity_pattern": {}
            }

    async def _generate_psychological_profile(
        self,
        user_id: UUID,
        social_media_data: Dict[SocialMediaPlatform, Dict[str, Any]]
    ) -> PsychologicalProfile:
        """
        Generar perfil psicológico basado en datos de redes sociales
        
        Args:
            user_id: ID del usuario
            social_media_data: Datos de redes sociales
            
        Returns:
            Perfil psicológico generado
        """
        logger.info("Generating psychological profile", user_id=str(user_id))

        try:
            # Verificar que hay datos suficientes
            total_posts = sum(
                len(data.get("posts", []))
                for data in social_media_data.values()
            )
            
            if total_posts == 0:
                raise InsufficientDataError(
                    "No posts found for analysis",
                    error_code="INSUFFICIENT_DATA",
                    details={"platforms": len(social_media_data)}
                )
            
            # Usar analizador avanzado
            analysis = await asyncio.wait_for(
                self._analyzer.analyze_social_media_data(social_media_data),
                timeout=config.analysis_timeout
            )
            
            # Extraer datos del análisis
            personality_traits = analysis.get("personality_traits", {})
            sentiment_analysis = analysis.get("sentiment_analysis", {})
            behavioral_patterns = analysis.get("behavioral_patterns", [])
            confidence_score = analysis.get("confidence_score", 0.5)
            
            # Determinar estado emocional
            overall_sentiment = sentiment_analysis.get("overall_sentiment", "neutral")
            emotional_stability = 1.0 - abs(sentiment_analysis.get("average_score", 0.0))
            stress_level = 0.3 if overall_sentiment == "negative" else 0.1
            
            # Identificar fortalezas y factores de riesgo
            strengths = []
            risk_factors = []
            
            if personality_traits.get("extraversion", 0.5) > 0.7:
                strengths.append("High social engagement")
            if personality_traits.get("conscientiousness", 0.5) > 0.7:
                strengths.append("Organized and disciplined")
            
            if sentiment_analysis.get("sentiment_distribution", {}).get("negative", 0) > 0.3:
                risk_factors.append("High negative sentiment detected")
            if personality_traits.get("neuroticism", 0.5) > 0.7:
                risk_factors.append("Elevated neuroticism levels")
            
            # Generar recomendaciones
            recommendations = []
            if overall_sentiment == "negative":
                recommendations.append("Consider seeking support if negative feelings persist")
            if personality_traits.get("neuroticism", 0.5) > 0.6:
                recommendations.append("Practice stress management techniques")
            if total_posts > 100:
                recommendations.append("Maintain healthy balance between online and offline activities")
            
            # Generar perfil
            profile = PsychologicalProfile(
                user_id=user_id,
                personality_traits=personality_traits,
                emotional_state={
                    "overall_sentiment": overall_sentiment,
                    "emotional_stability": max(0.0, min(1.0, emotional_stability)),
                    "stress_level": max(0.0, min(1.0, stress_level))
                },
                behavioral_patterns=behavioral_patterns,
                risk_factors=risk_factors,
                strengths=strengths,
                recommendations=recommendations,
                confidence_score=max(config.min_confidence_score, confidence_score)
            )

            logger.info(
                "Psychological profile generated",
                profile_id=str(profile.id),
                confidence_score=profile.confidence_score
            )
            return profile
            
        except asyncio.TimeoutError:
            raise AnalysisTimeoutError(
                "Analysis timeout",
                error_code="ANALYSIS_TIMEOUT",
                details={"timeout": config.analysis_timeout}
            )
        except InsufficientDataError:
            raise
        except Exception as e:
            logger.error("Error generating profile", error=str(e), user_id=str(user_id))
            raise ProfileGenerationError(
                f"Failed to generate profile: {str(e)}",
                error_code="PROFILE_GENERATION_ERROR",
                details={"error": str(e)}
            )

    async def _generate_validation_report(
        self,
        validation_id: UUID,
        social_media_data: Dict[SocialMediaPlatform, Dict[str, Any]],
        profile: PsychologicalProfile
    ) -> ValidationReport:
        """
        Generar reporte de validación
        
        Args:
            validation_id: ID de la validación
            social_media_data: Datos de redes sociales
            profile: Perfil psicológico
            
        Returns:
            Reporte generado
        """
        logger.info("Generating validation report", validation_id=str(validation_id))

        try:
            # Análisis avanzado para el reporte
            analysis = await self._analyzer.analyze_social_media_data(social_media_data)
            
            # Generar resumen ejecutivo
            total_posts = sum(len(data.get("posts", [])) for data in social_media_data.values())
            sentiment = analysis.get("sentiment_analysis", {})
            overall_sentiment = sentiment.get("overall_sentiment", "neutral")
            
            summary = f"""
Análisis psicológico completado exitosamente.

RESUMEN EJECUTIVO:
- Plataformas analizadas: {len(social_media_data)}
- Total de publicaciones analizadas: {total_posts}
- Confianza del análisis: {profile.confidence_score * 100:.1f}%

PERFIL PSICOLÓGICO:
- Sentimiento general: {overall_sentiment}
- Estabilidad emocional: {profile.emotional_state.get('emotional_stability', 0):.1%}
- Nivel de estrés: {profile.emotional_state.get('stress_level', 0):.1%}

HALLAZGOS:
- Fortalezas identificadas: {len(profile.strengths)}
- Factores de riesgo: {len(profile.risk_factors)}
- Patrones de comportamiento: {len(profile.behavioral_patterns)}
            """.strip()

            # Generar insights por plataforma
            social_media_insights = {}
            for platform, data in social_media_data.items():
                posts = data.get("posts", [])
                interactions = data.get("interactions", {})
                total_engagement = (
                    interactions.get("total_likes", 0) +
                    interactions.get("total_comments", 0) +
                    interactions.get("total_shares", 0)
                )
                engagement_rate = total_engagement / len(posts) if posts else 0
                
                social_media_insights[platform.value] = {
                    "post_count": len(posts),
                    "engagement_rate": engagement_rate,
                    "total_engagement": total_engagement,
                    "key_insights": [
                        "Active user" if len(posts) > 20 else "Moderate activity",
                        "High engagement" if engagement_rate > 50 else "Normal engagement"
                    ]
                }
            
            # Análisis temporal
            timeline_analysis = {}
            if social_media_data:
                # Analizar tendencia de actividad
                all_posts = []
                for data in social_media_data.values():
                    all_posts.extend(data.get("posts", []))
                
                if all_posts:
                    post_dates = [
                        datetime.fromisoformat(p.get("created_at", datetime.utcnow().isoformat()))
                        for p in all_posts
                    ]
                    post_dates.sort()
                    
                    if len(post_dates) > 1:
                        recent_posts = [d for d in post_dates if d > datetime.utcnow() - timedelta(days=30)]
                        timeline_analysis = {
                            "activity_trend": "increasing" if len(recent_posts) > len(post_dates) / 2 else "stable",
                            "peak_activity": "evening",  # Simplificado
                            "posts_last_30_days": len(recent_posts)
                        }
            
            # Análisis de sentimientos del reporte
            sentiment_analysis = {
                "overall_sentiment": overall_sentiment,
                "sentiment_distribution": sentiment.get("sentiment_distribution", {}),
                "average_score": sentiment.get("average_score", 0.0)
            }
            
            # Análisis de contenido (simplificado)
            content_analysis = {
                "total_posts": total_posts,
                "content_types": ["text", "images", "videos"],  # Simplificado
                "topics": ["general"]  # Simplificado
            }
            
            # Patrones de interacción
            total_interactions = sum(
                sum(data.get("interactions", {}).values())
                for data in social_media_data.values()
            )
            avg_engagement = total_interactions / total_posts if total_posts > 0 else 0
            
            interaction_patterns = {
                "engagement_level": "high" if avg_engagement > 50 else "medium" if avg_engagement > 20 else "low",
                "interaction_frequency": "regular",
                "average_engagement": avg_engagement
            }

            report = ValidationReport(
                validation_id=validation_id,
                summary=summary,
                detailed_analysis={
                    "personality": profile.personality_traits,
                    "emotional_state": profile.emotional_state,
                    "behavioral_patterns": profile.behavioral_patterns,
                    "confidence_score": profile.confidence_score
                },
                social_media_insights=social_media_insights,
                timeline_analysis=timeline_analysis,
                sentiment_analysis=sentiment_analysis,
                content_analysis=content_analysis,
                interaction_patterns=interaction_patterns
            )

            logger.info("Validation report generated", report_id=str(report.id))
            return report
            
        except Exception as e:
            logger.error("Error generating report", error=str(e), validation_id=str(validation_id))
            raise ReportGenerationError(
                f"Failed to generate report: {str(e)}",
                error_code="REPORT_GENERATION_ERROR",
                details={"error": str(e)}
            )

