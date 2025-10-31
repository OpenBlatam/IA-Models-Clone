from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging
from core.models import (
from core.engine import FacebookPostsEngine
from services.analytics_service import AnalyticsService
from infrastructure.repositories import FacebookPostRepository
from infrastructure.cache import CacheService
                from core.models import PostMetrics
from typing import Any, List, Dict, Optional
import asyncio
"""
📋 Application Use Cases - Casos de Uso de Aplicación
====================================================

Casos de uso que implementan la lógica de negocio para Facebook Posts.
Siguen los principios de Clean Architecture y Domain-Driven Design.
"""


    FacebookPost, PostRequest, PostResponse, PostStatus, 
    ContentType, AudienceType, OptimizationLevel, QualityTier,
    FacebookPostFactory
)

logger = logging.getLogger(__name__)

# ===== BASE USE CASE =====

class UseCase(ABC):
    """Clase base para todos los casos de uso."""
    
    def __init__(self) -> Any:
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Ejecutar el caso de uso. Debe ser implementado por subclases."""
        pass
    
    def _validate_input(self, data: Dict[str, Any]) -> bool:
        """Validar datos de entrada."""
        return True
    
    def _log_execution(self, method: str, **kwargs):
        """Registrar ejecución del caso de uso."""
        self.logger.info(f"Executing {method} with params: {kwargs}")

# ===== GENERATE POST USE CASE =====

class GeneratePostUseCase(UseCase):
    """
    Caso de uso para generar posts de Facebook.
    
    Maneja la lógica de negocio para la generación de posts,
    incluyendo validaciones, optimizaciones y analytics.
    """
    
    def __init__(self, engine: FacebookPostsEngine):
        
    """__init__ function."""
super().__init__()
        self.engine = engine
    
    async def execute(self, request: Union[PostRequest, Dict[str, Any]]) -> PostResponse:
        """
        Ejecutar generación de post.
        
        Args:
            request: PostRequest o diccionario con parámetros
            
        Returns:
            PostResponse con el resultado de la generación
        """
        self._log_execution("GeneratePostUseCase.execute", request=str(request)[:100])
        
        try:
            # Validar entrada
            if isinstance(request, dict):
                request = PostRequest.from_dict(request)
            
            self._validate_generation_request(request)
            
            # Ejecutar generación
            response = await self.engine.generate_post(request)
            
            # Validar resultado
            if response.success and response.post:
                self._validate_generated_post(response.post)
                
                # Aplicar reglas de negocio adicionales
                await self._apply_business_rules(response.post, request)
            
            self.logger.info(f"Post generation completed: {response.success}")
            return response
            
        except Exception as e:
            self.logger.error(f"Post generation failed: {e}")
            return PostResponse(
                success=False,
                error=str(e),
                processing_time=0.0
            )
    
    async def _validate_generation_request(self, request: PostRequest) -> None:
        """Validar request de generación."""
        if not request.topic or len(request.topic.strip()) == 0:
            raise ValueError("Topic cannot be empty")
        
        if len(request.topic) > 200:
            raise ValueError("Topic too long (max 200 characters)")
        
        if request.length and (request.length < 10 or request.length > 5000):
            raise ValueError("Length must be between 10 and 5000 characters")
        
        # Validar combinaciones de contenido y audiencia
        self._validate_content_audience_combination(request.content_type, request.audience_type)
    
    def _validate_content_audience_combination(self, content_type: ContentType, audience_type: AudienceType) -> None:
        """Validar combinación de tipo de contenido y audiencia."""
        invalid_combinations = [
            (ContentType.TECHNICAL, AudienceType.GENERAL),
            (ContentType.PROMOTIONAL, AudienceType.STUDENTS),
        ]
        
        if (content_type, audience_type) in invalid_combinations:
            raise ValueError(f"Invalid combination: {content_type.value} content for {audience_type.value} audience")
    
    def _validate_generated_post(self, post: FacebookPost) -> None:
        """Validar post generado."""
        if not post.content or len(post.content.strip()) == 0:
            raise ValueError("Generated post content cannot be empty")
        
        if len(post.content) > 5000:
            raise ValueError("Generated post too long (max 5000 characters)")
        
        # Verificar que el post tenga al menos 10 palabras
        word_count = post.get_word_count()
        if word_count < 10:
            raise ValueError(f"Generated post too short: {word_count} words (min 10)")
    
    async def _apply_business_rules(self, post: FacebookPost, request: PostRequest) -> None:
        """Aplicar reglas de negocio adicionales."""
        # Regla: Posts promocionales deben incluir hashtags
        if request.content_type == ContentType.PROMOTIONAL and not post.hashtags:
            self.logger.warning("Promotional post without hashtags")
        
        # Regla: Posts técnicos deben tener cierta longitud mínima
        if request.content_type == ContentType.TECHNICAL and post.get_word_count() < 50:
            self.logger.warning("Technical post might be too short")
        
        # Regla: Posts para audiencia profesional deben tener tono apropiado
        if request.audience_type == AudienceType.PROFESSIONALS:
            # Verificar tono profesional (implementación simplificada)
            professional_keywords = ['business', 'professional', 'industry', 'strategy', 'management']
            content_lower = post.content.lower()
            professional_score = sum(1 for keyword in professional_keywords if keyword in content_lower)
            
            if professional_score < 1:
                self.logger.warning("Post for professionals might lack professional tone")

# ===== ANALYZE POST USE CASE =====

class AnalyzePostUseCase(UseCase):
    """
    Caso de uso para analizar posts existentes.
    
    Proporciona análisis detallado de posts incluyendo métricas,
    recomendaciones y insights de engagement.
    """
    
    def __init__(self, analytics_service: AnalyticsService, post_repository: FacebookPostRepository):
        
    """__init__ function."""
super().__init__()
        self.analytics_service = analytics_service
        self.post_repository = post_repository
    
    async def execute(self, post_id: str) -> Dict[str, Any]:
        """
        Ejecutar análisis de post.
        
        Args:
            post_id: ID del post a analizar
            
        Returns:
            Diccionario con resultados del análisis
        """
        self._log_execution("AnalyzePostUseCase.execute", post_id=post_id)
        
        try:
            # Obtener post
            post = await self.post_repository.get_by_id(post_id)
            if not post:
                raise ValueError(f"Post not found: {post_id}")
            
            # Realizar análisis
            analysis_result = await self._perform_analysis(post)
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(post, analysis_result)
            
            # Actualizar post con métricas si no las tiene
            if not post.metrics and 'metrics' in analysis_result:
                metrics = PostMetrics(**analysis_result['metrics'])
                post.add_metrics(metrics)
                await self.post_repository.save(post)
            
            result = {
                'success': True,
                'post_id': post_id,
                'analysis': analysis_result,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Post analysis completed for {post_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Post analysis failed for {post_id}: {e}")
            return {
                'success': False,
                'post_id': post_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _perform_analysis(self, post: FacebookPost) -> Dict[str, Any]:
        """Realizar análisis del post."""
        # Análisis básico
        basic_analysis = {
            'word_count': post.get_word_count(),
            'character_count': post.get_character_count(),
            'reading_time': post.get_reading_time(),
            'hashtag_count': len(post.hashtags),
            'mention_count': len(post.mentions),
            'url_count': len(post.urls),
            'content_type': post.content_type.value,
            'audience_type': post.audience_type.value
        }
        
        # Análisis avanzado usando el servicio de analytics
        advanced_analysis = await self.analytics_service.analyze_post(post.to_dict())
        
        # Combinar análisis
        combined_analysis = {
            'basic': basic_analysis,
            'advanced': advanced_analysis,
            'quality_tier': post.quality_tier.value if post.quality_tier else None,
            'optimization_level': post.optimization_level.value
        }
        
        return combined_analysis
    
    def _generate_recommendations(self, post: FacebookPost, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generar recomendaciones basadas en el análisis."""
        recommendations = []
        
        # Recomendaciones basadas en longitud
        word_count = analysis['basic']['word_count']
        if word_count < 20:
            recommendations.append({
                'type': 'length',
                'priority': 'high',
                'message': 'Consider adding more content to increase engagement',
                'suggestion': 'Aim for 50-100 words for better reach'
            })
        elif word_count > 300:
            recommendations.append({
                'type': 'length',
                'priority': 'medium',
                'message': 'Post might be too long for social media',
                'suggestion': 'Consider breaking into multiple posts or using bullet points'
            })
        
        # Recomendaciones basadas en hashtags
        hashtag_count = analysis['basic']['hashtag_count']
        if hashtag_count == 0:
            recommendations.append({
                'type': 'hashtags',
                'priority': 'high',
                'message': 'No hashtags found',
                'suggestion': 'Add 3-5 relevant hashtags to increase discoverability'
            })
        elif hashtag_count > 10:
            recommendations.append({
                'type': 'hashtags',
                'priority': 'medium',
                'message': 'Too many hashtags',
                'suggestion': 'Limit to 5-7 hashtags for better readability'
            })
        
        # Recomendaciones basadas en tipo de contenido
        if post.content_type == ContentType.PROMOTIONAL and hashtag_count < 3:
            recommendations.append({
                'type': 'promotional',
                'priority': 'medium',
                'message': 'Promotional posts benefit from strategic hashtagging',
                'suggestion': 'Add industry-specific and trending hashtags'
            })
        
        # Recomendaciones basadas en métricas de calidad
        if post.metrics:
            if post.metrics.engagement_score < 0.6:
                recommendations.append({
                    'type': 'engagement',
                    'priority': 'high',
                    'message': 'Low engagement potential detected',
                    'suggestion': 'Consider adding questions, polls, or calls-to-action'
                })
            
            if post.metrics.readability_score < 0.7:
                recommendations.append({
                    'type': 'readability',
                    'priority': 'medium',
                    'message': 'Content might be difficult to read',
                    'suggestion': 'Use shorter sentences and simpler language'
                })
        
        return recommendations

# ===== APPROVE POST USE CASE =====

class ApprovePostUseCase(UseCase):
    """
    Caso de uso para aprobar posts.
    
    Maneja el flujo de aprobación incluyendo validaciones
    y actualización de estado.
    """
    
    def __init__(self, post_repository: FacebookPostRepository, cache_service: CacheService):
        
    """__init__ function."""
super().__init__()
        self.post_repository = post_repository
        self.cache_service = cache_service
    
    async def execute(self, post_id: str, approver_id: str, comments: Optional[str] = None) -> Dict[str, Any]:
        """
        Ejecutar aprobación de post.
        
        Args:
            post_id: ID del post a aprobar
            approver_id: ID del usuario que aprueba
            comments: Comentarios opcionales de aprobación
            
        Returns:
            Diccionario con resultado de la aprobación
        """
        self._log_execution("ApprovePostUseCase.execute", post_id=post_id, approver_id=approver_id)
        
        try:
            # Obtener post
            post = await self.post_repository.get_by_id(post_id)
            if not post:
                raise ValueError(f"Post not found: {post_id}")
            
            # Validar que el post puede ser aprobado
            self._validate_approval(post)
            
            # Aprobar post
            post.approve()
            
            # Añadir metadata de aprobación
            post.metadata.update({
                'approved_by': approver_id,
                'approved_at': datetime.now().isoformat(),
                'approval_comments': comments
            })
            
            # Guardar cambios
            await self.post_repository.save(post)
            
            # Invalidar cache
            await self.cache_service.delete(f"post:{post_id}")
            
            result = {
                'success': True,
                'post_id': post_id,
                'status': post.status.value,
                'approved_by': approver_id,
                'approved_at': post.metadata['approved_at'],
                'message': 'Post approved successfully'
            }
            
            self.logger.info(f"Post {post_id} approved by {approver_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Post approval failed for {post_id}: {e}")
            return {
                'success': False,
                'post_id': post_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_approval(self, post: FacebookPost) -> None:
        """Validar que el post puede ser aprobado."""
        if post.status != PostStatus.PENDING:
            raise ValueError(f"Cannot approve post in status: {post.status.value}")
        
        # Verificar que el post tiene contenido
        if not post.content or len(post.content.strip()) == 0:
            raise ValueError("Cannot approve post without content")
        
        # Verificar que el post cumple con estándares mínimos
        if post.get_word_count() < 10:
            raise ValueError("Cannot approve post with less than 10 words")
        
        # Verificar métricas de calidad si están disponibles
        if post.metrics and post.metrics.overall_score < 0.5:
            raise ValueError(f"Cannot approve post with low quality score: {post.metrics.overall_score}")

# ===== PUBLISH POST USE CASE =====

class PublishPostUseCase(UseCase):
    """
    Caso de uso para publicar posts.
    
    Maneja el flujo de publicación incluyendo validaciones
    finales y actualización de estado.
    """
    
    def __init__(self, post_repository: FacebookPostRepository, cache_service: CacheService):
        
    """__init__ function."""
super().__init__()
        self.post_repository = post_repository
        self.cache_service = cache_service
    
    async def execute(self, post_id: str, publisher_id: str, schedule_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Ejecutar publicación de post.
        
        Args:
            post_id: ID del post a publicar
            publisher_id: ID del usuario que publica
            schedule_time: Tiempo programado para publicación (opcional)
            
        Returns:
            Diccionario con resultado de la publicación
        """
        self._log_execution("PublishPostUseCase.execute", post_id=post_id, publisher_id=publisher_id)
        
        try:
            # Obtener post
            post = await self.post_repository.get_by_id(post_id)
            if not post:
                raise ValueError(f"Post not found: {post_id}")
            
            # Validar que el post puede ser publicado
            self._validate_publication(post)
            
            # Publicar post
            post.publish()
            
            # Añadir metadata de publicación
            post.metadata.update({
                'published_by': publisher_id,
                'published_at': datetime.now().isoformat(),
                'scheduled_time': schedule_time.isoformat() if schedule_time else None
            })
            
            # Guardar cambios
            await self.post_repository.save(post)
            
            # Invalidar cache
            await self.cache_service.delete(f"post:{post_id}")
            
            result = {
                'success': True,
                'post_id': post_id,
                'status': post.status.value,
                'published_by': publisher_id,
                'published_at': post.metadata['published_at'],
                'scheduled_time': post.metadata['scheduled_time'],
                'message': 'Post published successfully'
            }
            
            self.logger.info(f"Post {post_id} published by {publisher_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Post publication failed for {post_id}: {e}")
            return {
                'success': False,
                'post_id': post_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_publication(self, post: FacebookPost) -> None:
        """Validar que el post puede ser publicado."""
        if post.status != PostStatus.APPROVED:
            raise ValueError(f"Cannot publish post in status: {post.status.value}")
        
        # Verificar que el post está listo para publicación
        if not post.is_ready_for_publication():
            raise ValueError("Post is not ready for publication")
        
        # Verificar métricas mínimas
        if post.metrics and post.metrics.overall_score < 0.7:
            raise ValueError(f"Cannot publish post with low quality score: {post.metrics.overall_score}")

# ===== GET ANALYTICS USE CASE =====

class GetAnalyticsUseCase(UseCase):
    """
    Caso de uso para obtener analytics del sistema.
    
    Proporciona métricas y insights del sistema completo.
    """
    
    def __init__(self, engine: FacebookPostsEngine, analytics_service: AnalyticsService):
        
    """__init__ function."""
super().__init__()
        self.engine = engine
        self.analytics_service = analytics_service
    
    async def execute(self, analytics_type: str = 'system', filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ejecutar obtención de analytics.
        
        Args:
            analytics_type: Tipo de analytics ('system', 'posts', 'performance')
            filters: Filtros opcionales para los analytics
            
        Returns:
            Diccionario con analytics solicitados
        """
        self._log_execution("GetAnalyticsUseCase.execute", analytics_type=analytics_type)
        
        try:
            if analytics_type == 'system':
                return await self._get_system_analytics()
            elif analytics_type == 'posts':
                return await self._get_posts_analytics(filters)
            elif analytics_type == 'performance':
                return await self._get_performance_analytics(filters)
            else:
                raise ValueError(f"Unknown analytics type: {analytics_type}")
                
        except Exception as e:
            self.logger.error(f"Analytics retrieval failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _get_system_analytics(self) -> Dict[str, Any]:
        """Obtener analytics del sistema."""
        return await self.engine.get_system_analytics()
    
    async def _get_posts_analytics(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Obtener analytics de posts."""
        # Implementación simplificada
        return {
            'success': True,
            'type': 'posts',
            'filters': filters,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_performance_analytics(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Obtener analytics de performance."""
        # Implementación simplificada
        return {
            'success': True,
            'type': 'performance',
            'filters': filters,
            'timestamp': datetime.now().isoformat()
        }

# ===== USE CASE FACTORY =====

class UseCaseFactory:
    """Factory para crear casos de uso."""
    
    def __init__(
        self,
        engine: FacebookPostsEngine,
        analytics_service: AnalyticsService,
        post_repository: FacebookPostRepository,
        cache_service: CacheService
    ):
        
    """__init__ function."""
self.engine = engine
        self.analytics_service = analytics_service
        self.post_repository = post_repository
        self.cache_service = cache_service
    
    def create_generate_post_use_case(self) -> GeneratePostUseCase:
        """Crear caso de uso de generación de posts."""
        return GeneratePostUseCase(self.engine)
    
    def create_analyze_post_use_case(self) -> AnalyzePostUseCase:
        """Crear caso de uso de análisis de posts."""
        return AnalyzePostUseCase(self.analytics_service, self.post_repository)
    
    def create_approve_post_use_case(self) -> ApprovePostUseCase:
        """Crear caso de uso de aprobación de posts."""
        return ApprovePostUseCase(self.post_repository, self.cache_service)
    
    def create_publish_post_use_case(self) -> PublishPostUseCase:
        """Crear caso de uso de publicación de posts."""
        return PublishPostUseCase(self.post_repository, self.cache_service)
    
    def create_get_analytics_use_case(self) -> GetAnalyticsUseCase:
        """Crear caso de uso de obtención de analytics."""
        return GetAnalyticsUseCase(self.engine, self.analytics_service)

# ===== EXPORTS =====

__all__ = [
    # Base
    'UseCase',
    
    # Use Cases
    'GeneratePostUseCase',
    'AnalyzePostUseCase',
    'ApprovePostUseCase',
    'PublishPostUseCase',
    'GetAnalyticsUseCase',
    
    # Factory
    'UseCaseFactory'
] 