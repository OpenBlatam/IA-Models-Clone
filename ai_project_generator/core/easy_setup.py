"""
Easy Setup - Configuración fácil y rápida
=========================================

Helpers para configurar rápidamente el sistema sin complejidad.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI

from .microservices_config import MicroservicesConfig, DeploymentType
from .microservices_integration import setup_microservices_app

logger = logging.getLogger(__name__)


class EasyConfig:
    """Configuración simplificada con valores por defecto sensatos"""
    
    def __init__(
        self,
        # Básico
        enable_cache: bool = True,
        enable_workers: bool = False,
        enable_events: bool = False,
        enable_metrics: bool = True,
        
        # Avanzado (opcional)
        redis_url: Optional[str] = None,
        deployment_type: str = "standard",
        
        # Auto-detect
        auto_detect: bool = True
    ):
        """
        Configuración fácil con valores por defecto.
        
        Args:
            enable_cache: Habilitar cache (usa Redis si está disponible, sino in-memory)
            enable_workers: Habilitar workers asíncronos
            enable_events: Habilitar eventos (message broker)
            enable_metrics: Habilitar métricas Prometheus
            redis_url: URL de Redis (opcional, auto-detect si no se proporciona)
            deployment_type: Tipo de despliegue (standard, serverless)
            auto_detect: Auto-detectar servicios disponibles
        """
        self.enable_cache = enable_cache
        self.enable_workers = enable_workers
        self.enable_events = enable_events
        self.enable_metrics = enable_metrics
        self.redis_url = redis_url
        self.deployment_type = deployment_type
        self.auto_detect = auto_detect
    
    def to_microservices_config(self) -> Dict[str, Any]:
        """Convierte a configuración de microservicios"""
        config = {}
        
        # Cache
        if self.enable_cache:
            if self.redis_url:
                config["cache_backend"] = "redis"
                config["cache_url"] = self.redis_url
            elif self.auto_detect:
                # Intentar auto-detectar Redis
                try:
                    import redis
                    redis.Redis.from_url("redis://localhost:6379").ping()
                    config["cache_backend"] = "redis"
                    config["cache_url"] = "redis://localhost:6379"
                except:
                    config["cache_backend"] = "in_memory"
            else:
                config["cache_backend"] = "in_memory"
        else:
            config["cache_backend"] = "none"
        
        # Workers
        if self.enable_workers:
            if self.redis_url:
                config["worker_backend"] = "celery"
                config["worker_broker_url"] = self.redis_url
            else:
                config["worker_backend"] = "none"
        else:
            config["worker_backend"] = "none"
        
        # Events
        if self.enable_events:
            if self.redis_url:
                config["message_broker_type"] = "redis"
                config["message_broker_url"] = self.redis_url
            else:
                config["message_broker_type"] = "none"
        else:
            config["message_broker_type"] = "none"
        
        # Metrics
        config["prometheus_enabled"] = self.enable_metrics
        
        # Deployment
        config["deployment_type"] = self.deployment_type
        
        # Valores por defecto sensatos
        config.setdefault("structured_logging", True)
        config.setdefault("circuit_breaker_enabled", True)
        config.setdefault("retry_enabled", True)
        
        return config


def create_app_easy(
    title: str = "AI Project Generator API",
    version: str = "1.0.0",
    enable_cache: bool = True,
    enable_metrics: bool = True,
    **kwargs
) -> FastAPI:
    """
    Crea aplicación FastAPI de forma fácil.
    
    Args:
        title: Título de la API
        version: Versión de la API
        enable_cache: Habilitar cache (default: True)
        enable_metrics: Habilitar métricas (default: True)
        **kwargs: Opciones adicionales
    
    Returns:
        Aplicación FastAPI configurada
    
    Example:
        ```python
        from core.easy_setup import create_app_easy
        
        app = create_app_easy()
        # Listo para usar!
        ```
    """
    from ..api.app_factory import create_app
    
    # Configurar fácilmente
    easy_config = EasyConfig(
        enable_cache=enable_cache,
        enable_metrics=enable_metrics,
        **kwargs
    )
    
    # Aplicar configuración
    import os
    config = easy_config.to_microservices_config()
    for key, value in config.items():
        env_key = f"MICROSERVICES_{key.upper()}"
        if env_key not in os.environ:
            os.environ[env_key] = str(value)
    
    # Crear app
    app = create_app(
        base_output_dir=kwargs.get("base_output_dir", "generated_projects"),
        enable_continuous=kwargs.get("enable_continuous", True),
        title=title,
        version=version
    )
    
    logger.info("Easy setup completed successfully!")
    return app


def quick_start() -> FastAPI:
    """
    Inicio rápido con configuración mínima.
    
    Returns:
        Aplicación FastAPI lista para usar
    
    Example:
        ```python
        from core.easy_setup import quick_start
        
        app = quick_start()
        # ¡Listo en una línea!
        ```
    """
    return create_app_easy()


# Presets comunes
def create_app_development() -> FastAPI:
    """Preset para desarrollo (todo habilitado, in-memory si no hay Redis)"""
    return create_app_easy(
        enable_cache=True,
        enable_metrics=True,
        enable_workers=False,
        enable_events=False
    )


def create_app_production(redis_url: str = "redis://localhost:6379") -> FastAPI:
    """Preset para producción (todo habilitado con Redis)"""
    return create_app_easy(
        enable_cache=True,
        enable_metrics=True,
        enable_workers=True,
        enable_events=True,
        redis_url=redis_url
    )


def create_app_serverless() -> FastAPI:
    """Preset para serverless (mínimo, sin workers)"""
    return create_app_easy(
        enable_cache=False,  # Serverless no necesita cache persistente
        enable_metrics=True,
        enable_workers=False,
        enable_events=False,
        deployment_type="serverless"
    )















