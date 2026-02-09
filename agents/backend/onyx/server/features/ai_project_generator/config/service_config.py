"""
Service Configuration - Configuración de servicios
==================================================

Configuración específica de servicios de negocio.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class ServiceConfig(BaseSettings):
    """Configuración de servicios"""
    
    # Repository
    repository_type: str = "continuous_generator"  # continuous_generator, memory
    
    # Generation
    default_async_generation: bool = True
    batch_parallel: bool = True
    batch_max_size: int = 50
    
    # Validation
    validation_enabled: bool = True
    validation_strict: bool = False
    
    # Export
    export_formats: list = ["zip", "tar", "tar.gz"]
    export_default_format: str = "zip"
    
    class Config:
        env_prefix = "SERVICE_"
        case_sensitive = False


def get_service_config() -> ServiceConfig:
    """Obtiene configuración de servicios"""
    return ServiceConfig()















