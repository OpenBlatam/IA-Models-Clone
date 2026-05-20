"""
Unified Settings for the ads feature.

This module consolidates configuration settings from:
- config.py (basic settings)
- optimized_config.py (production settings)

Provides a clean, hierarchical configuration system with environment-specific defaults.
"""

from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from functools import lru_cache


class Environment(str, Enum):
    """Environment types for configuration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Basic settings for Onyx ads functionality."""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # API Settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database Settings
    database_url: str = Field(
        default="sqlite:///./onyx.db",
        description="Database connection URL"
    )
    
    # Storage Settings
    storage_path: str = Field(
        default="./storage",
        description="Path for file storage"
    )
    
    # LangChain Settings
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for LLM operations"
    )
    openai_model: str = "gpt-4-turbo-preview"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2000
    
    # Vector Store Settings
    vector_store_path: str = Field(
        default="./vector_store",
        description="Path for vector store data"
    )
    
    # Cache Settings
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds (1 hour)"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


class OptimizedSettings(BaseSettings):
    """Production-ready optimized settings for Onyx ads functionality."""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # API Settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_requests: int = 1000
    max_requests_jitter: int = 100
    
    # Database Settings
    database_url: str = Field(
        default="sqlite+aiosqlite:///./onyx.db",
        description="Database connection URL with async support"
    )
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    
    # Redis Settings
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    redis_max_connections: int = 50
    redis_socket_timeout: int = 5
    redis_socket_connect_timeout: int = 5
    
    # Storage Settings
    storage_path: str = Field(
        default="./storage",
        description="Path for file storage"
    )
    storage_url: str = Field(
        default="/storage",
        description="URL path for storage access"
    )
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: List[str] = ["jpg", "jpeg", "png", "gif", "webp"]
    
    # Image Processing Settings
    max_image_size: int = 2048
    max_image_size_bytes: int = 10 * 1024 * 1024  # 10MB
    jpeg_quality: int = 85
    png_optimize: bool = True
    image_cache_ttl: int = 86400  # 24 hours
    
    # LLM Settings
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for LLM operations"
    )
    openai_model: str = "gpt-4-turbo-preview"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2000
    openai_timeout: int = 60
    openai_max_retries: int = 3
    
    # Vector Store Settings
    vector_store_path: str = Field(
        default="./vector_store",
        description="Path for vector store data"
    )
    embedding_model: str = "text-embedding-3-small"
    vector_dimension: int = 1536
    
    # Cache Settings
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 10000
    cache_cleanup_interval: int = 300  # 5 minutes
    
    # Rate Limiting
    rate_limits: Dict[str, int] = Field(
        default={
            "ads_generation": 100,  # per hour
            "background_removal": 50,  # per hour
            "analytics_tracking": 1000,  # per hour
            "file_upload": 20,  # per hour
        },
        description="Rate limits for different operations"
    )
    
    # Security Settings
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # Monitoring and Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )
    prometheus_enabled: bool = True
    health_check_interval: int = 30
    
    # Performance Settings
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    connection_timeout: int = 10
    keepalive_timeout: int = 5
    
    # Background Tasks
    background_task_workers: int = 4
    task_queue_size: int = 1000
    task_timeout: int = 300  # 5 minutes
    
    # File Processing
    chunk_size: int = 8192
    max_memory_usage: int = 512 * 1024 * 1024  # 512MB
    temp_file_cleanup_interval: int = 3600  # 1 hour
    
    # Analytics
    analytics_enabled: bool = True
    analytics_retention_days: int = 90
    analytics_batch_size: int = 100
    analytics_flush_interval: int = 60  # 1 minute
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        env_prefix = "ADS_"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached basic settings instance."""
    return Settings()


@lru_cache()
def get_optimized_settings() -> OptimizedSettings:
    """Get cached optimized settings instance."""
    return OptimizedSettings()


# Export default settings instances
settings = get_settings()
optimized_settings = get_optimized_settings() 