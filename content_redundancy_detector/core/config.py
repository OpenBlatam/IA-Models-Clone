"""
Centralized Configuration Management
Handles all application configuration with environment variable support
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "Content Redundancy Detector"
    app_version: str = "2.0.0"
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # CORS
    cors_origins: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_cache_url: str = os.getenv("REDIS_CACHE_URL", "redis://localhost:6379/1")
    redis_rate_limit_url: str = os.getenv("REDIS_RATE_LIMIT_URL", "redis://localhost:6379/2")
    
    # Database
    database_url: Optional[str] = os.getenv("DATABASE_URL")
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Rate Limiting
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    rate_limit_per_hour: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    rate_limit_per_day: int = int(os.getenv("RATE_LIMIT_PER_DAY", "10000"))
    
    # Cache
    cache_default_ttl: int = int(os.getenv("CACHE_DEFAULT_TTL", "3600"))
    cache_max_size: int = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    
    # Serverless
    serverless_mode: bool = os.getenv("SERVERLESS_MODE", "false").lower() == "true"
    
    # API Gateway
    api_gateway_enabled: bool = os.getenv("API_GATEWAY_ENABLED", "false").lower() == "true"
    
    # Observability
    opentelemetry_enabled: bool = os.getenv("OPENTELEMETRY_ENABLED", "true").lower() == "true"
    otlp_endpoint: str = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
    
    # Monitoring
    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()