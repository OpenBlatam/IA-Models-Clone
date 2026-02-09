"""
Settings - Configuraciones principales del sistema
"""

from pydantic import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Configuraciones principales de la aplicación"""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8020
    debug: bool = False

    # Database
    database_url: str = "sqlite:///./suno_clone.db"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Security
    secret_key: str = "change-me-in-production"
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"

    class Config:
        env_file = ".env"
        case_sensitive = False

