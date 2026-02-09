"""
Settings - Configuración
=======================

Configuración del sistema ultra refactorizado real.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Configuración de la aplicación."""
    
    # Configuración de la aplicación
    APP_NAME: str = "AI History Comparison API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Configuración del servidor
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Configuración de CORS
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="ALLOWED_ORIGINS"
    )
    
    # Configuración de base de datos
    DATABASE_URL: str = Field(default="sqlite:///./ai_history_comparison.db", env="DATABASE_URL")
    DATABASE_PATH: str = Field(default="ai_history_comparison.db", env="DATABASE_PATH")
    
    # Configuración de logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configuración de análisis
    MAX_CONTENT_LENGTH: int = Field(default=10000, env="MAX_CONTENT_LENGTH")
    ANALYSIS_TIMEOUT: int = Field(default=30, env="ANALYSIS_TIMEOUT")
    
    # Configuración de API externas
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    
    # Configuración de límites
    MAX_ENTRIES_PER_REQUEST: int = Field(default=100, env="MAX_ENTRIES_PER_REQUEST")
    MAX_COMPARISONS_PER_REQUEST: int = Field(default=50, env="MAX_COMPARISONS_PER_REQUEST")
    
    # Configuración de caché
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hora
    ENABLE_CACHE: bool = Field(default=True, env="ENABLE_CACHE")
    
    # Configuración de monitoreo
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_INTERVAL: int = Field(default=60, env="METRICS_INTERVAL")  # segundos
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Instancia global de configuración
settings = Settings()




