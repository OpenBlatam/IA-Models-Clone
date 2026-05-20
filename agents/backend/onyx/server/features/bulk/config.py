"""
Configuración de la API usando variables de entorno
====================================================
"""

import os
from typing import List
from functools import lru_cache

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings
    except ImportError:
        from pydantic import BaseModel as BaseSettings


class Settings(BaseSettings):
    """Configuración de la aplicación."""
    
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    cors_origins: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_ttl: int = int(os.getenv("REDIS_TTL", "3600"))
    
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./bul_api.db")
    
    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
    
    monitoring_enabled: bool = os.getenv("MONITORING_ENABLED", "true").lower() == "true"
    health_check_interval: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    
    bul_enabled: bool = os.getenv("BUL_ENABLED", "true").lower() == "true"
    bul_simulation_mode: bool = os.getenv("BUL_SIMULATION_MODE", "false").lower() == "true"
    
    log_file: str = os.getenv("LOG_FILE", "bul_api.log")
    log_json: bool = os.getenv("LOG_JSON", "false").lower() == "true"
    log_structured: bool = os.getenv("LOG_STRUCTURED", "false").lower() == "true"
    
    otel_enabled: bool = os.getenv("OTEL_ENABLED", "false").lower() == "true"
    otel_service_name: str = os.getenv("OTEL_SERVICE_NAME", "bul-api")
    otel_exporter_otlp_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from: str = os.getenv("SMTP_FROM", "noreply@bul-api.com")
    
    webhook_url: str = os.getenv("WEBHOOK_URL", "")
    webhook_secret: str = os.getenv("WEBHOOK_SECRET", "")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Obtiene la configuración (cached)."""
    return Settings()


settings = get_settings()

