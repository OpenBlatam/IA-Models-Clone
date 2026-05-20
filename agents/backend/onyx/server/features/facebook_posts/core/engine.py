from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import asyncio
import time
import uuid
import logging
import json
from .models import (
from optimization.base import (
from services.ai_service import AIService
from services.analytics_service import AnalyticsService
from infrastructure.cache import CacheService
from infrastructure.repositories import FacebookPostRepository
        import hashlib
from typing import Any, List, Dict, Optional
"""
🚀 Core Engine - Motor Principal Refactorizado
=============================================

Motor principal refactorizado que integra todos los componentes del sistema
de Facebook Posts con Clean Architecture y optimizaciones.
"""


    FacebookPost, PostRequest, PostResponse, PostStatus, 
    ContentType, AudienceType, OptimizationLevel, QualityTier,
    FacebookPostFactory, PostMetrics
)
    OptimizationPipeline, OptimizationContext, OptimizationResult,
    OptimizerFactory
)

# Configurar logging
logger = logging.getLogger(__name__)

class FacebookPostsEngine:
    """
    Motor principal refactorizado para Facebook Posts.
    
    Integra generación de contenido, optimizaciones, analytics y gestión
    de posts con Clean Architecture y patrones de diseño modernos.
    """
    
    def __init__(
        self,
        ai_service: AIService,
        analytics_service: AnalyticsService,
        cache_service: CacheService,
        post_repository: FacebookPostRepository,
        optimization_pipeline: Optional[OptimizationPipeline] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        
    """__init__ function."""
self.ai_service = ai_service
        self.analytics_service = analytics_service
        self.cache_service = cache_service
        self.post_repository = post_repository
        self.optimization_pipeline = optimization_pipeline or OptimizationPipeline()
        self.config = config or {}
        
        # Métricas del engine
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Configurar optimizadores por defecto
        self._setup_default_optimizers()
        
        logger.info(f"FacebookPostsEngine initialized with {len(self.optimization_pipeline.optimizers)} optimizers")
    
    def _setup_default_optimizers(self) -> None:
        """Configurar optimizadores por defecto."""
        default_optimizers = self.config.get('default_optimizers', [
            'performance',
            'quality', 
            'analytics',
            'model_selection'
        ])
        
        for optimizer_name in default_optimizers:
            try:
                optimizer = OptimizerFactory.create(optimizer_name)
                self.optimization_pipeline.add_optimizer(optimizer)
                logger.info(f"Added default optimizer: {optimizer_name}")
            except Exception as e:
                logger.warning(f"Failed to add optimizer {optimizer_name}: {e}")
    
    async def generate_post(self, request: Union[PostRequest, Dict[str, Any]]) -> PostResponse:
        """
        Generar un post de Facebook con optimizaciones completas.
        
        Args:
            request: PostRequest o diccionario con parámetros
            
        Returns:
            PostResponse con el post generado y métricas
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Convertir request si es necesario
            if isinstance(request, dict):
                request = PostRequest.from_dict(request)
            
            # Validar request
            self._validate_request(request)
            
            # Verificar cache
            cache_key = self._generate_cache_key(request)
            cached_result = await self.cache_service.get(cache_key)
            
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.info(f"Cache hit for request {request_id}")
                return PostResponse(
                    success=True,
                    post=cached_result,
                    processing_time=time.time() - start_time,
                    optimizations_applied=['cache']
                )
            
            self.stats['cache_misses'] += 1
            
            # Generar contenido base
            content = await self._generate_base_content(request)
            
            # Crear post inicial
            post = FacebookPostFactory.create_from_request(request, content)
            
            # Aplicar optimizaciones
            optimized_post = await self._apply_optimizations(post, request)
            
            # Analizar post
            analytics = await self._analyze_post(optimized_post)
            
            # Añadir métricas al post
            if analytics and 'metrics' in analytics:
                metrics = PostMetrics(**analytics['metrics'])
                optimized_post.add_metrics(metrics)
            
            # Guardar en repositorio
            await self.post_repository.save(optimized_post)
            
            # Guardar en cache
            await self.cache_service.set(cache_key, optimized_post, ttl=3600)
            
            # Actualizar estadísticas
            processing_time = time.time() - start_time
            self._update_stats(True, processing_time)
            
            logger.info(f"Successfully generated post {optimized_post.id} in {processing_time:.2f}s")
            
            return PostResponse(
                success=True,
                post=optimized_post,
                processing_time=processing_time,
                optimizations_applied=self._get_applied_optimizations(),
                analytics=analytics
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(False, processing_time)
            
            logger.error(f"Failed to generate post for request {request_id}: {e}")
            
            return PostResponse(
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    async def _generate_base_content(self, request: PostRequest) -> str:
        """Generar contenido base usando el servicio de IA."""
        try:
            content = await self.ai_service.generate_content(request.to_dict())
            return content
        except Exception as e:
            logger.error(f"Failed to generate base content: {e}")
            raise
    
    async def _apply_optimizations(self, post: FacebookPost, request: PostRequest) -> FacebookPost:
        """Aplicar optimizaciones al post."""
        if not self.optimization_pipeline.optimizers:
            return post
        
        try:
            # Preparar datos para optimización
            optimization_data = {
                'post': post.to_dict(),
                'request': request.to_dict(),
                'content': post.content,
                'metadata': post.metadata
            }
            
            # Crear contexto de optimización
            context = OptimizationContext(
                request_id=post.id,
                user_id=request.metadata.get('user_id'),
                session_id=request.metadata.get('session_id'),
                metadata=request.metadata
            )
            
            # Ejecutar pipeline de optimización
            result = await self.optimization_pipeline.optimize(optimization_data, context)
            
            if result.success and 'post' in result.optimized_data:
                # Recrear post con datos optimizados
                optimized_post_data = result.optimized_data['post']
                optimized_post = FacebookPost.from_dict(optimized_post_data)
                
                # Actualizar optimizations_applied en metadata
                optimized_post.metadata['optimizations_applied'] = result.optimizations_applied
                optimized_post.metadata['optimization_metrics'] = result.metrics
                
                return optimized_post
            
            return post
            
        except Exception as e:
            logger.warning(f"Optimization failed, returning original post: {e}")
            return post
    
    async def _analyze_post(self, post: FacebookPost) -> Optional[Dict[str, Any]]:
        """Analizar post usando el servicio de analytics."""
        try:
            analytics = await self.analytics_service.analyze_post(post.to_dict())
            return analytics
        except Exception as e:
            logger.warning(f"Analytics failed: {e}")
            return None
    
    async def _validate_request(self, request: PostRequest) -> None:
        """Validar request de generación."""
        if not request.topic or len(request.topic.strip()) == 0:
            raise ValueError("Topic cannot be empty")
        
        if request.length and request.length <= 0:
            raise ValueError("Length must be positive")
        
        if request.length and request.length > 5000:
            raise ValueError("Length cannot exceed 5000 characters")
    
    def _generate_cache_key(self, request: PostRequest) -> str:
        """Generar clave de cache para el request."""
        request_data = request.to_dict()
        request_hash = hashlib.md5(json.dumps(request_data, sort_keys=True).encode()).hexdigest()
        return f"facebook_post:{request_hash}"
    
    def _get_applied_optimizations(self) -> List[str]:
        """Obtener lista de optimizaciones aplicadas."""
        return [opt.name for opt in self.optimization_pipeline.optimizers if opt.is_enabled]
    
    def _update_stats(self, success: bool, processing_time: float) -> None:
        """Actualizar estadísticas del engine."""
        self.stats['total_requests'] += 1
        self.stats['total_processing_time'] += processing_time
        self.stats['avg_processing_time'] = self.stats['total_processing_time'] / self.stats['total_requests']
        
        if success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
    
    # ===== MÉTODOS DE GESTIÓN DE POSTS =====
    
    async def get_post(self, post_id: str) -> Optional[FacebookPost]:
        """Obtener un post por ID."""
        try:
            return await self.post_repository.get_by_id(post_id)
        except Exception as e:
            logger.error(f"Failed to get post {post_id}: {e}")
            return None
    
    async def list_posts(
        self, 
        status: Optional[PostStatus] = None,
        content_type: Optional[ContentType] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[FacebookPost]:
        """Listar posts con filtros."""
        try:
            filters = {}
            if status:
                filters['status'] = status
            if content_type:
                filters['content_type'] = content_type
            
            return await self.post_repository.list(filters, limit, offset)
        except Exception as e:
            logger.error(f"Failed to list posts: {e}")
            return []
    
    async def update_post(self, post_id: str, updates: Dict[str, Any]) -> Optional[FacebookPost]:
        """Actualizar un post."""
        try:
            post = await self.post_repository.get_by_id(post_id)
            if not post:
                return None
            
            # Aplicar actualizaciones
            if 'content' in updates:
                post.update_content(updates['content'])
            
            if 'status' in updates:
                new_status = PostStatus(updates['status'])
                if new_status == PostStatus.APPROVED:
                    post.approve()
                elif new_status == PostStatus.PUBLISHED:
                    post.publish()
                elif new_status == PostStatus.REJECTED:
                    post.reject(updates.get('rejection_reason', ''))
                elif new_status == PostStatus.ARCHIVED:
                    post.archive()
            
            # Guardar cambios
            await self.post_repository.save(post)
            
            # Invalidar cache
            await self.cache_service.delete(f"post:{post_id}")
            
            return post
            
        except Exception as e:
            logger.error(f"Failed to update post {post_id}: {e}")
            return None
    
    async def delete_post(self, post_id: str) -> bool:
        """Eliminar un post."""
        try:
            success = await self.post_repository.delete(post_id)
            if success:
                # Invalidar cache
                await self.cache_service.delete(f"post:{post_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to delete post {post_id}: {e}")
            return False
    
    # ===== MÉTODOS DE ANALYTICS =====
    
    async def get_post_analytics(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Obtener analytics de un post específico."""
        try:
            post = await self.get_post(post_id)
            if not post:
                return None
            
            return await self.analytics_service.analyze_post(post.to_dict())
        except Exception as e:
            logger.error(f"Failed to get analytics for post {post_id}: {e}")
            return None
    
    async def get_system_analytics(self) -> Dict[str, Any]:
        """Obtener analytics del sistema completo."""
        try:
            # Analytics del engine
            engine_analytics = {
                'requests': self.stats,
                'cache_performance': {
                    'hit_rate': self.stats['cache_hits'] / max(self.stats['total_requests'], 1),
                    'miss_rate': self.stats['cache_misses'] / max(self.stats['total_requests'], 1)
                }
            }
            
            # Analytics de optimizadores
            optimization_analytics = self.optimization_pipeline.get_pipeline_metrics()
            
            # Analytics de posts
            posts_analytics = await self._get_posts_analytics()
            
            return {
                'engine': engine_analytics,
                'optimization': optimization_analytics,
                'posts': posts_analytics,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get system analytics: {e}")
            return {}
    
    async def _get_posts_analytics(self) -> Dict[str, Any]:
        """Obtener analytics de posts."""
        try:
            all_posts = await self.post_repository.list({}, limit=1000)
            
            if not all_posts:
                return {}
            
            # Estadísticas por status
            status_counts = {}
            content_type_counts = {}
            quality_tier_counts = {}
            total_posts = len(all_posts)
            
            for post in all_posts:
                # Status counts
                status = post.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Content type counts
                content_type = post.content_type.value
                content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
                
                # Quality tier counts
                if post.quality_tier:
                    quality_tier = post.quality_tier.value
                    quality_tier_counts[quality_tier] = quality_tier_counts.get(quality_tier, 0) + 1
            
            return {
                'total_posts': total_posts,
                'status_distribution': {k: v/total_posts for k, v in status_counts.items()},
                'content_type_distribution': {k: v/total_posts for k, v in content_type_counts.items()},
                'quality_tier_distribution': {k: v/total_posts for k, v in quality_tier_counts.items()},
                'recent_posts': len([p for p in all_posts if (datetime.now() - p.created_at).days <= 7])
            }
        except Exception as e:
            logger.error(f"Failed to get posts analytics: {e}")
            return {}
    
    # ===== MÉTODOS DE CONFIGURACIÓN =====
    
    def add_optimizer(self, optimizer_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Añadir un optimizador al pipeline."""
        try:
            optimizer = OptimizerFactory.create(optimizer_name, config)
            self.optimization_pipeline.add_optimizer(optimizer)
            logger.info(f"Added optimizer: {optimizer_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add optimizer {optimizer_name}: {e}")
            return False
    
    def remove_optimizer(self, optimizer_name: str) -> bool:
        """Remover un optimizador del pipeline."""
        try:
            self.optimization_pipeline.remove_optimizer(optimizer_name)
            logger.info(f"Removed optimizer: {optimizer_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove optimizer {optimizer_name}: {e}")
            return False
    
    def get_optimizer(self, optimizer_name: str):
        """Obtener un optimizador del pipeline."""
        return self.optimization_pipeline.get_optimizer(optimizer_name)
    
    def update_optimizer_config(self, optimizer_name: str, config: Dict[str, Any]) -> bool:
        """Actualizar configuración de un optimizador."""
        try:
            optimizer = self.get_optimizer(optimizer_name)
            if optimizer:
                optimizer.update_config(config)
                logger.info(f"Updated config for optimizer: {optimizer_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update config for optimizer {optimizer_name}: {e}")
            return False
    
    # ===== MÉTODOS DE UTILIDAD =====
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del engine."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Resetear estadísticas del engine."""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        logger.info("Engine stats reset")
    
    def get_config(self) -> Dict[str, Any]:
        """Obtener configuración del engine."""
        return {
            'config': self.config,
            'optimizers': [opt.name for opt in self.optimization_pipeline.optimizers],
            'enabled_optimizers': [opt.name for opt in self.optimization_pipeline.optimizers if opt.is_enabled]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del engine."""
        try:
            # Verificar servicios
            ai_health = await self.ai_service.health_check()
            analytics_health = await self.analytics_service.health_check()
            cache_health = await self.cache_service.health_check()
            repo_health = await self.post_repository.health_check()
            
            # Verificar optimizadores
            optimizer_health = {}
            for optimizer in self.optimization_pipeline.optimizers:
                optimizer_health[optimizer.name] = {
                    'enabled': optimizer.is_enabled,
                    'metrics': optimizer.get_metrics()
                }
            
            return {
                'status': 'healthy',
                'services': {
                    'ai_service': ai_health,
                    'analytics_service': analytics_health,
                    'cache_service': cache_health,
                    'repository': repo_health
                },
                'optimizers': optimizer_health,
                'stats': self.stats,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# ===== FACTORY FUNCTION =====

async def create_facebook_posts_engine(
    ai_service: AIService,
    analytics_service: AnalyticsService,
    cache_service: CacheService,
    post_repository: FacebookPostRepository,
    config: Optional[Dict[str, Any]] = None
) -> FacebookPostsEngine:
    """
    Factory function para crear una instancia del engine.
    
    Args:
        ai_service: Servicio de IA
        analytics_service: Servicio de analytics
        cache_service: Servicio de cache
        post_repository: Repositorio de posts
        config: Configuración opcional
        
    Returns:
        Instancia configurada del FacebookPostsEngine
    """
    engine = FacebookPostsEngine(
        ai_service=ai_service,
        analytics_service=analytics_service,
        cache_service=cache_service,
        post_repository=post_repository,
        config=config
    )
    
    # Verificar salud inicial
    health = await engine.health_check()
    if health['status'] != 'healthy':
        logger.warning(f"Engine health check failed: {health}")
    
    return engine

# ===== EXPORTS =====

__all__ = [
    'FacebookPostsEngine',
    'create_facebook_posts_engine'
] 