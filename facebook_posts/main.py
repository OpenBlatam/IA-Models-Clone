from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from core.models import (
from core.engine import FacebookPostsEngine, create_facebook_posts_engine
from application import (
from optimization.base import OptimizationPipeline, OptimizerFactory
from services.ai_service import AIService
from services.analytics_service import AnalyticsService
from infrastructure.cache import CacheService
from infrastructure.repositories import FacebookPostRepository
        import traceback
from typing import Any, List, Dict, Optional
"""
🚀 Facebook Posts - Main Entry Point
===================================

Punto de entrada principal para el sistema de Facebook Posts refactorizado.
Proporciona una interfaz limpia y unificada para todas las funcionalidades.
"""


# Import core components
    FacebookPost, PostRequest, PostResponse, PostStatus, 
    ContentType, AudienceType, OptimizationLevel, QualityTier,
    FacebookPostFactory
)

# Import application layer
    GeneratePostUseCase, AnalyzePostUseCase, ApprovePostUseCase,
    PublishPostUseCase, GetAnalyticsUseCase, UseCaseFactory
)

# Import optimization system

# Import services (these would be implemented in the services module)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FacebookPostsSystem:
    """
    Sistema principal de Facebook Posts refactorizado.
    
    Proporciona una interfaz unificada para todas las funcionalidades
    del sistema, incluyendo generación, optimización, análisis y gestión.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.engine: Optional[FacebookPostsEngine] = None
        self.use_case_factory: Optional[UseCaseFactory] = None
        self.initialized = False
        
        logger.info("FacebookPostsSystem initialized")
    
    async def initialize(self) -> None:
        """Inicializar el sistema completo."""
        try:
            logger.info("Initializing Facebook Posts System...")
            
            # Initialize services
            ai_service = AIService(self.config.get('ai', {}))
            analytics_service = AnalyticsService(self.config.get('analytics', {}))
            cache_service = CacheService(self.config.get('cache', {}))
            post_repository = FacebookPostRepository(self.config.get('repository', {}))
            
            # Create engine
            self.engine = await create_facebook_posts_engine(
                ai_service=ai_service,
                analytics_service=analytics_service,
                cache_service=cache_service,
                post_repository=post_repository,
                config=self.config.get('engine', {})
            )
            
            # Create use case factory
            self.use_case_factory = UseCaseFactory(
                engine=self.engine,
                analytics_service=analytics_service,
                post_repository=post_repository,
                cache_service=cache_service
            )
            
            self.initialized = True
            logger.info("Facebook Posts System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Facebook Posts System: {e}")
            raise
    
    def _ensure_initialized(self) -> None:
        """Verificar que el sistema esté inicializado."""
        if not self.initialized:
            raise RuntimeError("Facebook Posts System not initialized. Call initialize() first.")
    
    # ===== MAIN OPERATIONS =====
    
    async def generate_post(self, request: Union[PostRequest, Dict[str, Any]]) -> PostResponse:
        """
        Generar un post de Facebook.
        
        Args:
            request: PostRequest o diccionario con parámetros
            
        Returns:
            PostResponse con el post generado
        """
        self._ensure_initialized()
        
        use_case = self.use_case_factory.create_generate_post_use_case()
        return await use_case.execute(request)
    
    async def analyze_post(self, post_id: str) -> Dict[str, Any]:
        """
        Analizar un post existente.
        
        Args:
            post_id: ID del post a analizar
            
        Returns:
            Diccionario con resultados del análisis
        """
        self._ensure_initialized()
        
        use_case = self.use_case_factory.create_analyze_post_use_case()
        return await use_case.execute(post_id)
    
    async def approve_post(self, post_id: str, approver_id: str, comments: Optional[str] = None) -> Dict[str, Any]:
        """
        Aprobar un post.
        
        Args:
            post_id: ID del post a aprobar
            approver_id: ID del usuario que aprueba
            comments: Comentarios opcionales
            
        Returns:
            Diccionario con resultado de la aprobación
        """
        self._ensure_initialized()
        
        use_case = self.use_case_factory.create_approve_post_use_case()
        return await use_case.execute(post_id, approver_id, comments)
    
    async def publish_post(self, post_id: str, publisher_id: str, schedule_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Publicar un post.
        
        Args:
            post_id: ID del post a publicar
            publisher_id: ID del usuario que publica
            schedule_time: Tiempo programado para publicación
            
        Returns:
            Diccionario con resultado de la publicación
        """
        self._ensure_initialized()
        
        use_case = self.use_case_factory.create_publish_post_use_case()
        return await use_case.execute(post_id, publisher_id, schedule_time)
    
    async def get_analytics(self, analytics_type: str = 'system', filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Obtener analytics del sistema.
        
        Args:
            analytics_type: Tipo de analytics ('system', 'posts', 'performance')
            filters: Filtros opcionales
            
        Returns:
            Diccionario con analytics solicitados
        """
        self._ensure_initialized()
        
        use_case = self.use_case_factory.create_get_analytics_use_case()
        return await use_case.execute(analytics_type, filters)
    
    # ===== POST MANAGEMENT =====
    
    async def get_post(self, post_id: str) -> Optional[FacebookPost]:
        """Obtener un post por ID."""
        self._ensure_initialized()
        return await self.engine.get_post(post_id)
    
    async def list_posts(
        self, 
        status: Optional[PostStatus] = None,
        content_type: Optional[ContentType] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[FacebookPost]:
        """Listar posts con filtros."""
        self._ensure_initialized()
        return await self.engine.list_posts(status, content_type, limit, offset)
    
    async def update_post(self, post_id: str, updates: Dict[str, Any]) -> Optional[FacebookPost]:
        """Actualizar un post."""
        self._ensure_initialized()
        return await self.engine.update_post(post_id, updates)
    
    async def delete_post(self, post_id: str) -> bool:
        """Eliminar un post."""
        self._ensure_initialized()
        return await self.engine.delete_post(post_id)
    
    # ===== OPTIMIZATION MANAGEMENT =====
    
    def add_optimizer(self, optimizer_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Añadir un optimizador al sistema."""
        self._ensure_initialized()
        return self.engine.add_optimizer(optimizer_name, config)
    
    def remove_optimizer(self, optimizer_name: str) -> bool:
        """Remover un optimizador del sistema."""
        self._ensure_initialized()
        return self.engine.remove_optimizer(optimizer_name)
    
    def get_optimizer(self, optimizer_name: str):
        """Obtener un optimizador del sistema."""
        self._ensure_initialized()
        return self.engine.get_optimizer(optimizer_name)
    
    def update_optimizer_config(self, optimizer_name: str, config: Dict[str, Any]) -> bool:
        """Actualizar configuración de un optimizador."""
        self._ensure_initialized()
        return self.engine.update_optimizer_config(optimizer_name, config)
    
    # ===== SYSTEM INFORMATION =====
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema."""
        self._ensure_initialized()
        return self.engine.get_stats()
    
    def get_config(self) -> Dict[str, Any]:
        """Obtener configuración del sistema."""
        self._ensure_initialized()
        return self.engine.get_config()
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del sistema."""
        self._ensure_initialized()
        return await self.engine.health_check()
    
    def get_available_optimizers(self) -> List[str]:
        """Obtener lista de optimizadores disponibles."""
        return OptimizerFactory.get_available()
    
    # ===== UTILITY METHODS =====
    
    async def create_sample_request(self, topic: str = "Digital Marketing Tips") -> PostRequest:
        """Crear un request de ejemplo."""
        return PostRequest(
            topic=topic,
            audience_type=AudienceType.PROFESSIONALS,
            content_type=ContentType.EDUCATIONAL,
            tone="professional",
            optimization_level=OptimizationLevel.STANDARD
        )
    
    def create_sample_post(self) -> FacebookPost:
        """Crear un post de ejemplo."""
        return FacebookPostFactory.create_sample_post()
    
    async def quick_generate(self, topic: str, audience: str = "professionals", content_type: str = "educational") -> PostResponse:
        """
        Generación rápida de post con parámetros mínimos.
        
        Args:
            topic: Tema del post
            audience: Tipo de audiencia
            content_type: Tipo de contenido
            
        Returns:
            PostResponse con el post generado
        """
        request = PostRequest(
            topic=topic,
            audience_type=AudienceType(audience),
            content_type=ContentType(content_type)
        )
        
        return await self.generate_post(request)

# ===== CONVENIENCE FUNCTIONS =====

async def create_facebook_posts_system(config: Optional[Dict[str, Any]] = None) -> FacebookPostsSystem:
    """
    Crear e inicializar el sistema de Facebook Posts.
    
    Args:
        config: Configuración opcional del sistema
        
    Returns:
        Sistema inicializado listo para usar
    """
    system = FacebookPostsSystem(config)
    await system.initialize()
    return system

async def quick_generate_post(
    topic: str,
    audience: str = "professionals",
    content_type: str = "educational",
    config: Optional[Dict[str, Any]] = None
) -> PostResponse:
    """
    Generación rápida de post sin necesidad de inicializar el sistema completo.
    
    Args:
        topic: Tema del post
        audience: Tipo de audiencia
        content_type: Tipo de contenido
        config: Configuración opcional
        
    Returns:
        PostResponse con el post generado
    """
    system = await create_facebook_posts_system(config)
    return await system.quick_generate(topic, audience, content_type)

# ===== DEMO FUNCTION =====

async def run_demo():
    """Ejecutar demo del sistema refactorizado."""
    print("🚀 Facebook Posts System - Refactored Demo")
    print("=" * 50)
    
    try:
        # Crear sistema
        system = await create_facebook_posts_system()
        
        # Health check
        health = await system.health_check()
        print(f"✅ System Health: {health['status']}")
        
        # Generar post de ejemplo
        print("\n📝 Generating sample post...")
        response = await system.quick_generate(
            "AI in Modern Business",
            "professionals",
            "educational"
        )
        
        if response.success:
            post = response.post
            print(f"✅ Post generated successfully!")
            print(f"   ID: {post.id}")
            print(f"   Content: {post.content[:100]}...")
            print(f"   Status: {post.status.value}")
            print(f"   Quality Tier: {post.quality_tier.value if post.quality_tier else 'N/A'}")
            print(f"   Processing Time: {response.processing_time:.2f}s")
            print(f"   Optimizations Applied: {response.optimizations_applied}")
            
            # Analizar post
            print("\n📊 Analyzing post...")
            analysis = await system.analyze_post(post.id)
            if analysis['success']:
                print(f"✅ Analysis completed!")
                recommendations = analysis.get('recommendations', [])
                print(f"   Recommendations: {len(recommendations)}")
                for rec in recommendations[:3]:  # Show first 3
                    print(f"   - {rec['message']}")
            
            # Obtener analytics del sistema
            print("\n📈 Getting system analytics...")
            analytics = await system.get_analytics('system')
            if analytics.get('success'):
                print(f"✅ Analytics retrieved!")
                engine_stats = analytics.get('engine', {}).get('requests', {})
                print(f"   Total Requests: {engine_stats.get('total_requests', 0)}")
                print(f"   Success Rate: {engine_stats.get('successful_requests', 0) / max(engine_stats.get('total_requests', 1), 1):.1%}")
                print(f"   Avg Processing Time: {engine_stats.get('avg_processing_time', 0):.3f}s")
        
        else:
            print(f"❌ Post generation failed: {response.error}")
        
        # Mostrar optimizadores disponibles
        print(f"\n🔧 Available Optimizers: {len(system.get_available_optimizers())}")
        for optimizer in system.get_available_optimizers():
            print(f"   - {optimizer}")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        traceback.print_exc()

# ===== MAIN ENTRY POINT =====

if __name__ == "__main__":
    # Ejecutar demo si se ejecuta directamente
    asyncio.run(run_demo())

# ===== EXPORTS =====

__all__ = [
    'FacebookPostsSystem',
    'create_facebook_posts_system',
    'quick_generate_post',
    'run_demo'
] 