from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Ultra-Optimized Configuration v10
Production-ready settings with maximum performance
"""



class Settings(BaseSettings):
    """Ultra-optimized application settings"""
    
    # Application
    app_name: str = Field(default="Ultra-Fast SEO Service v10", env="APP_NAME")
    version: str = Field(default="10.0.0", env="VERSION")
    environment: str = Field(default="production", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    
    # API
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    docs_url: Optional[str] = Field(default="/docs", env="DOCS_URL")
    redoc_url: Optional[str] = Field(default="/redoc", env="REDOC_URL")
    openapi_url: Optional[str] = Field(default="/openapi.json", env="OPENAPI_URL")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_credentials: bool = Field(default=True, env="CORS_CREDENTIALS")
    cors_methods: List[str] = Field(default=["*"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")
    
    # Performance
    enable_compression: bool = Field(default=True, env="ENABLE_COMPRESSION")
    compression_threshold: int = Field(default=1024, env="COMPRESSION_THRESHOLD")
    max_connections: int = Field(default=100, env="MAX_CONNECTIONS")
    max_keepalive_connections: int = Field(default=20, env="MAX_KEEPALIVE_CONNECTIONS")
    
    # HTTP Client
    http_timeout: float = Field(default=30.0, env="HTTP_TIMEOUT")
    http_retry_attempts: int = Field(default=3, env="HTTP_RETRY_ATTEMPTS")
    http_user_agent: str = Field(default="UltraFastSEO/10.0", env="HTTP_USER_AGENT")
    
    # Cache
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    cache_maxsize: int = Field(default=10000, env="CACHE_MAXSIZE")
    cache_compression: bool = Field(default=True, env="CACHE_COMPRESSION")
    
    # Redis
    redis_enabled: bool = Field(default=False, env="REDIS_ENABLED")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_ttl: int = Field(default=3600, env="REDIS_TTL")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Monitoring
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    
    # Security
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    
    # SEO Analysis
    max_urls_per_batch: int = Field(default=50, env="MAX_URLS_PER_BATCH")
    analysis_timeout: float = Field(default=60.0, env="ANALYSIS_TIMEOUT")
    follow_redirects: bool = Field(default=True, env="FOLLOW_REDIRECTS")
    
    # Database (if needed)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # Background Tasks
    celery_enabled: bool = Field(default=False, env="CELERY_ENABLED")
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # File Storage
    upload_dir: str = Field(default="/tmp/uploads", env="UPLOAD_DIR")
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    
    # External Services
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    datadog_api_key: Optional[str] = Field(default=None, env="DATADOG_API_KEY")
    
    # Development
    reload: bool = Field(default=False, env="RELOAD")
    reload_dirs: List[str] = Field(default=[], env="RELOAD_DIRS")
    
    @dataclass
class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Convenience access
settings = get_settings()


def validate_settings():
    """Validate critical settings"""
    if settings.workers < 1:
        raise ValueError("Workers must be at least 1")
    
    if settings.port < 1 or settings.port > 65535:
        raise ValueError("Port must be between 1 and 65535")
    
    if settings.http_timeout < 1:
        raise ValueError("HTTP timeout must be at least 1 second")
    
    if settings.cache_ttl < 1:
        raise ValueError("Cache TTL must be at least 1 second")
    
    if settings.max_urls_per_batch < 1 or settings.max_urls_per_batch > 100:
        raise ValueError("Max URLs per batch must be between 1 and 100")


# Validate settings on import
validate_settings()


def get_database_config() -> dict:
    """
    Get database configuration
    
    Returns:
        dict: Database configuration
    """
    return {
        "url": settings.database_url,
        "pool_size": settings.database_pool_size,
        "max_overflow": settings.database_max_overflow,
        "echo": settings.debug
    }


def get_redis_config() -> dict:
    """
    Get Redis configuration
    
    Returns:
        dict: Redis configuration
    """
    return {
        "url": settings.redis_url,
        "max_connections": settings.max_connections,
        "encoding": "utf-8",
        "decode_responses": True,
        "retry_on_timeout": True,
        "socket_keepalive": True,
        "health_check_interval": 30
    }


def get_logging_config() -> dict:
    """
    Get logging configuration
    
    Returns:
        dict: Logging configuration
    """
    return {
        "level": settings.log_level,
        "format": settings.log_format,
        "file": settings.log_file,
        "max_size": settings.max_file_size,  # Assuming max_file_size is used for log_max_size
        "backup_count": 5  # Assuming log_backup_count is 5
    }


def get_cors_config() -> dict:
    """
    Get CORS configuration
    
    Returns:
        dict: CORS configuration
    """
    return {
        "allow_origins": settings.cors_origins,
        "allow_credentials": settings.cors_credentials,
        "allow_methods": settings.cors_methods,
        "allow_headers": settings.cors_headers
    }


def is_development() -> bool:
    """
    Check if running in development mode
    
    Returns:
        bool: True if development mode
    """
    return settings.environment == "development"


def is_production() -> bool:
    """
    Check if running in production mode
    
    Returns:
        bool: True if production mode
    """
    return settings.environment == "production"


def is_staging() -> bool:
    """
    Check if running in staging mode
    
    Returns:
        bool: True if staging mode
    """
    return settings.environment == "staging" 