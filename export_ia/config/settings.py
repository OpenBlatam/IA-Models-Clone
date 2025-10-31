"""
Configuración principal de la aplicación
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración de la aplicación."""
    
    # API
    api_title: str = "Export IA API"
    api_version: str = "2.0.0"
    api_description: str = "API para exportación de documentos con IA"
    debug: bool = False
    
    # Base de datos
    database_url: str = "sqlite:///./export_ia.db"
    database_echo: bool = False
    
    # Archivos
    exports_dir: str = "./exports"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: list = [".pdf", ".docx", ".html", ".md", ".rtf", ".txt", ".json", ".xml"]
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/export_ia.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Tareas
    max_concurrent_tasks: int = 10
    task_timeout: int = 300  # 5 minutos
    task_retry_attempts: int = 3
    
    # Calidad
    default_quality_level: str = "professional"
    enable_quality_validation: bool = True
    enable_content_enhancement: bool = True
    
    # Seguridad
    secret_key: str = "your-secret-key-here"
    access_token_expire_minutes: int = 30
    
    # CORS
    cors_origins: list = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Instancia global de configuración
settings = Settings()


def get_settings() -> Settings:
    """Obtener configuración de la aplicación."""
    return settings




