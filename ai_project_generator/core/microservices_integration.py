"""
Microservices Integration - Integración completa de características de microservicios
====================================================================================

Integra todas las características avanzadas de microservicios:
- Middleware avanzado
- Circuit breakers
- Retry logic
- Redis cache
- Prometheus metrics
- API Gateway
- Message brokers
- Async workers
- Serverless optimizations
- OAuth2 security
"""

import logging
from fastapi import FastAPI
from typing import Optional

from .microservices_config import get_microservices_config
from .advanced_middleware import setup_advanced_middleware
from .prometheus_metrics import (
    PrometheusMiddleware,
    get_metrics,
    get_metrics_content_type
)
from .redis_client import get_redis_client
from .circuit_breaker import get_circuit_breaker
from .retry_logic import RetryConfig
from .api_gateway import get_api_gateway_client
from .message_broker import get_message_broker
from .async_workers import get_worker_manager
from .serverless_optimizer import get_serverless_optimizer
from .oauth2_security import get_oauth2_manager

logger = logging.getLogger(__name__)


class MicroservicesIntegration:
    """Integración completa de características de microservicios"""
    
    def __init__(self):
        self.config = get_microservices_config()
        self.redis_client = None
        self.worker_manager = None
        self.api_gateway_client = None
        self.message_broker = None
        self.serverless_optimizer = None
        self.oauth2_manager = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Inicializa todos los componentes"""
        # Redis
        if self.config.cache_backend.value != "none":
            try:
                self.redis_client = get_redis_client()
                logger.info("Redis client initialized")
            except Exception as e:
                logger.warning(f"Redis client not available: {e}")
        
        # Workers
        if self.config.worker_backend.value != "none":
            try:
                self.worker_manager = get_worker_manager()
                logger.info(f"Worker manager initialized ({self.config.worker_backend.value})")
            except Exception as e:
                logger.warning(f"Worker manager not available: {e}")
        
        # API Gateway
        if self.config.api_gateway_type.value != "none":
            try:
                self.api_gateway_client = get_api_gateway_client()
                logger.info(f"API Gateway client initialized ({self.config.api_gateway_type.value})")
            except Exception as e:
                logger.warning(f"API Gateway client not available: {e}")
        
        # Message Broker
        if self.config.message_broker_type.value != "none":
            try:
                self.message_broker = get_message_broker()
                logger.info(f"Message broker initialized ({self.config.message_broker_type.value})")
            except Exception as e:
                logger.warning(f"Message broker not available: {e}")
        
        # Serverless Optimizer
        if self.config.deployment_type == "serverless":
            try:
                self.serverless_optimizer = get_serverless_optimizer()
                logger.info("Serverless optimizer initialized")
            except Exception as e:
                logger.warning(f"Serverless optimizer not available: {e}")
        
        # OAuth2
        if self.config.oauth2_enabled:
            try:
                self.oauth2_manager = get_oauth2_manager()
                logger.info("OAuth2 manager initialized")
            except Exception as e:
                logger.warning(f"OAuth2 manager not available: {e}")
    
    def setup_app(self, app: FastAPI) -> FastAPI:
        """
        Configura la aplicación FastAPI con todas las características de microservicios.
        
        Args:
            app: Aplicación FastAPI
        
        Returns:
            Aplicación configurada
        """
        # Middleware avanzado
        setup_advanced_middleware(app)
        
        # Prometheus metrics
        if self.config.prometheus_enabled:
            app.add_middleware(PrometheusMiddleware)
            
            # Endpoint de métricas
            @app.get("/metrics", tags=["monitoring"])
            async def metrics():
                """Endpoint de métricas Prometheus"""
                from fastapi.responses import Response
                return Response(
                    content=get_metrics(),
                    media_type=get_metrics_content_type()
                )
        
        # Serverless optimizations
        if self.config.deployment_type == "serverless" and self.serverless_optimizer:
            if self.config.serverless_provider == "aws_lambda":
                # Crear handler Lambda
                lambda_handler = self.serverless_optimizer.create_lambda_handler(app)
                app.state.lambda_handler = lambda_handler
        
        logger.info("Microservices integration configured successfully")
        return app
    
    async def register_with_api_gateway(
        self,
        service_name: str,
        service_url: str,
        routes: list
    ) -> bool:
        """Registra el servicio con el API Gateway"""
        if not self.api_gateway_client:
            return False
        
        return await self.api_gateway_client.register_service(
            service_name,
            service_url,
            routes
        )
    
    async def publish_event(
        self,
        topic: str,
        event: dict
    ) -> bool:
        """Publica un evento en el message broker"""
        if not self.message_broker:
            return False
        
        return await self.message_broker.publish(topic, event)
    
    def enqueue_task(
        self,
        task_func,
        *args,
        **kwargs
    ) -> Optional[str]:
        """Encola una tarea para ejecución asíncrona"""
        if not self.worker_manager:
            return None
        
        return self.worker_manager.enqueue_task(task_func, *args, **kwargs)
    
    async def get_cache(self, key: str):
        """Obtiene valor del cache"""
        if not self.redis_client:
            return None
        
        return await self.redis_client.aget(key)
    
    async def set_cache(
        self,
        key: str,
        value: any,
        ttl: Optional[int] = None
    ) -> bool:
        """Establece valor en cache"""
        if not self.redis_client:
            return False
        
        return await self.redis_client.aset(key, value, ttl)
    
    async def close(self):
        """Cierra todas las conexiones"""
        if self.redis_client:
            await self.redis_client.aclose()
        
        if self.api_gateway_client:
            await self.api_gateway_client.close()
        
        if self.message_broker:
            self.message_broker.close()


# Instancia global
_microservices_integration: Optional[MicroservicesIntegration] = None


def get_microservices_integration() -> MicroservicesIntegration:
    """Obtiene instancia global de integración de microservicios"""
    global _microservices_integration
    if _microservices_integration is None:
        _microservices_integration = MicroservicesIntegration()
    return _microservices_integration


def setup_microservices_app(app: FastAPI) -> FastAPI:
    """
    Función helper para configurar aplicación con microservicios.
    
    Usage:
        app = FastAPI()
        app = setup_microservices_app(app)
    """
    integration = get_microservices_integration()
    return integration.setup_app(app)















