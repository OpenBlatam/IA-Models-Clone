"""
App Configuration - Configuración de la aplicación
==================================================

Configuración específica de la aplicación FastAPI.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class AppConfig(BaseSettings):
    """Configuración de la aplicación"""
    
    # Básico
    title: str = "AI Project Generator API"
    version: str = "1.0.0"
    description: str = "API modular para generar proyectos de IA"
    
    # Servidor
    host: str = "0.0.0.0"
    port: int = 8020
    reload: bool = False
    
    # Directorios
    base_output_dir: str = "generated_projects"
    
    # Features
    enable_continuous: bool = True
    enable_cache: bool = True
    enable_metrics: bool = True
    enable_workers: bool = False
    enable_events: bool = False
    
    # CORS
    cors_origins: list = ["*"]
    cors_credentials: bool = True
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]
    
    class Config:
        env_prefix = "APP_"
        case_sensitive = False


def get_app_config() -> AppConfig:
    """Obtiene configuración de la aplicación"""
    return AppConfig()















