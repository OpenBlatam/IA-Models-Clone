"""
Configuración del sistema Addiction Recovery AI
"""

import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # Aplicación
    app_name: str = "Addiction Recovery AI"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8018"))
    
    # OpenAI
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Base de datos
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./recovery.db")
    
    # Seguridad
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # segundos
    
    # Notificaciones
    enable_notifications: bool = os.getenv("ENABLE_NOTIFICATIONS", "True").lower() == "true"
    
    # Cache
    redis_url: Optional[str] = os.getenv("REDIS_URL")
    cache_ttl: int = 3600  # 1 hora
    
    # CORS
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
    cors_allow_credentials: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "True").lower() == "true"
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    
    # Rate limiting
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    rate_limit_per_hour: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

