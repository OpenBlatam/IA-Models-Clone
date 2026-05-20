from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List
import os
from functools import lru_cache
from enum import Enum
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Optimized configuration for Onyx ads functionality with production-ready settings.
"""

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class OptimizedSettings(BaseSettings):
    """Optimized settings for Onyx ads functionality."""
    
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
    database_url: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./onyx.db")
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    
    # Redis Settings
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_max_connections: int = 50
    redis_socket_timeout: int = 5
    redis_socket_connect_timeout: int = 5
    
    # Storage Settings
    storage_path: str = os.getenv("STORAGE_PATH", "./storage")
    storage_url: str = os.getenv("STORAGE_URL", "/storage")
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: List[str] = ["jpg", "jpeg", "png", "gif", "webp"]
    
    # Image Processing Settings
    max_image_size: int = 2048
    max_image_size_bytes: int = 10 * 1024 * 1024  # 10MB
    jpeg_quality: int = 85
    png_optimize: bool = True
    image_cache_ttl: int = 86400  # 24 hours
    
    # LLM Settings
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = "gpt-4-turbo-preview"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2000
    openai_timeout: int = 60
    openai_max_retries: int = 3
    
    # Vector Store Settings
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    embedding_model: str = "text-embedding-3-small"
    vector_dimension: int = 1536
    
    # Cache Settings
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 10000
    cache_cleanup_interval: int = 300  # 5 minutes
    
    # Rate Limiting
    rate_limits: Dict[str, int] = {
        "ads_generation": 100,  # per hour
        "background_removal": 50,  # per hour
        "analytics_tracking": 1000,  # per hour
        "file_upload": 20,  # per hour
    }
    
    # Security Settings
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # Monitoring and Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN")
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
    
    # Model Settings
    llm_model: str = "gpt-4-turbo-preview"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    
    @dataclass
class Config:
        env_file = ".env"
        case_sensitive = True
        env_prefix = "ADS_"

@lru_cache()
def get_optimized_settings() -> OptimizedSettings:
    """Get cached optimized settings instance."""
    return OptimizedSettings()

def get_llm_config():
    """Get LLM configuration with optimized settings."""
    settings = get_optimized_settings()
    
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout,
        max_retries=settings.openai_max_retries
    )

def get_embeddings_config():
    """Get embeddings configuration with optimized settings."""
    settings = get_optimized_settings()
    
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        chunk_size=settings.chunk_size
    )

def get_redis_config():
    """Get Redis configuration with optimized settings."""
    settings = get_optimized_settings()
    
    return {
        "url": settings.redis_url,
        "max_connections": settings.redis_max_connections,
        "socket_timeout": settings.redis_socket_timeout,
        "socket_connect_timeout": settings.redis_socket_connect_timeout,
        "encoding": "utf-8",
        "decode_responses": True
    }

def get_database_config():
    """Get database configuration with optimized settings."""
    settings = get_optimized_settings()
    
    return {
        "url": settings.database_url,
        "pool_size": settings.database_pool_size,
        "max_overflow": settings.database_max_overflow,
        "pool_timeout": settings.database_pool_timeout,
        "pool_recycle": settings.database_pool_recycle,
        "echo": settings.debug
    }

# Export settings instance
settings = get_optimized_settings() 