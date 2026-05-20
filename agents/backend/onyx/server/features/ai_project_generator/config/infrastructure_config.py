"""
Infrastructure Configuration - Configuración de infraestructura
===============================================================

Configuración específica de servicios de infraestructura.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class InfrastructureConfig(BaseSettings):
    """Configuración de infraestructura"""
    
    # Cache
    cache_backend: str = "auto"  # auto, redis, memory, none
    cache_url: Optional[str] = None
    cache_ttl: int = 3600
    
    # Workers
    worker_backend: str = "auto"  # auto, celery, rq, none
    worker_broker_url: Optional[str] = None
    
    # Events
    event_backend: str = "auto"  # auto, redis, rabbitmq, kafka, none
    event_broker_url: Optional[str] = None
    
    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9090
    
    class Config:
        env_prefix = "INFRA_"
        case_sensitive = False


def get_infrastructure_config() -> InfrastructureConfig:
    """Obtiene configuración de infraestructura"""
    return InfrastructureConfig()















