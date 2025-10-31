from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
from enum import Enum
from functools import lru_cache
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Core Configuration Module
========================

Centralized application configuration using Pydantic Settings
with environment variable support and validation.
"""




class Environment(str, Enum):
    """Application environment types."""
    LOCAL = "local"
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AppSettings(BaseSettings):
    """Application settings."""
    
    # Basic app info
    name: str = Field(default="Product API", env="APP_NAME")
    version: str = Field(default="2.0.0", env="APP_VERSION")
    description: str = Field(default="Refactored Product Management API", env="APP_DESCRIPTION")
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")
    workers: int = Field(default=1, env="WORKERS")
    
    # API settings
    api_prefix: str = Field(default="/api/v2", env="API_PREFIX")
    docs_url: Optional[str] = Field(default="/docs", env="DOCS_URL")
    redoc_url: Optional[str] = Field(default="/redoc", env="REDOC_URL")
    openapi_url: Optional[str] = Field(default="/openapi.json", env="OPENAPI_URL")
    
    @validator('environment', pre=True)
    def validate_environment(cls, v) -> bool:
        if isinstance(v, str):
            return v.lower()
        return v
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        return self.environment in [Environment.DEVELOPMENT, Environment.LOCAL]


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    # Connection settings
    url: str = Field(default="postgresql+asyncpg://user:pass@localhost/products", env="DATABASE_URL")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")
    
    # Query settings
    echo: bool = Field(default=False, env="DB_ECHO")
    echo_pool: bool = Field(default=False, env="DB_ECHO_POOL")
    
    # Health check
    ping_timeout: float = Field(default=5.0, env="DB_PING_TIMEOUT")


class RedisSettings(BaseSettings):
    """Redis configuration."""
    
    # Connection
    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    timeout: float = Field(default=5.0, env="REDIS_TIMEOUT")
    retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")
    
    # Cache TTL settings (in seconds)
    default_ttl: int = Field(default=3600, env="CACHE_DEFAULT_TTL")  # 1 hour
    search_ttl: int = Field(default=300, env="CACHE_SEARCH_TTL")     # 5 minutes
    analytics_ttl: int = Field(default=600, env="CACHE_ANALYTICS_TTL")  # 10 minutes


class SecuritySettings(BaseSettings):
    """Security configuration."""
    
    # API Security
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    api_keys: List[str] = Field(default_factory=list, env="API_KEYS")
    secret_key: str = Field(default="change-me-in-production", env="SECRET_KEY")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    requests_per_minute: int = Field(default=100, env="REQUESTS_PER_MINUTE")
    requests_per_hour: int = Field(default=1000, env="REQUESTS_PER_HOUR")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    
    # Request limits
    max_request_size: int = Field(default=10_000_000, env="MAX_REQUEST_SIZE")  # 10MB
    max_bulk_operations: int = Field(default=100, env="MAX_BULK_OPERATIONS")


class ObservabilitySettings(BaseSettings):
    """Monitoring and observability configuration."""
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Metrics
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Tracing
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    
    # Error tracking
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    sentry_environment: Optional[str] = Field(default=None, env="SENTRY_ENVIRONMENT")
    
    # Performance
    slow_query_threshold: float = Field(default=1.0, env="SLOW_QUERY_THRESHOLD")
    enable_profiling: bool = Field(default=False, env="ENABLE_PROFILING")


class AISettings(BaseSettings):
    """AI and ML configuration."""
    
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=500, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    
    # Features
    enable_ai_descriptions: bool = Field(default=False, env="ENABLE_AI_DESCRIPTIONS")
    enable_content_moderation: bool = Field(default=False, env="ENABLE_CONTENT_MODERATION")
    ai_timeout: float = Field(default=10.0, env="AI_TIMEOUT")


class Settings(BaseSettings):
    """Main application settings container."""
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": False}
    
    # Sub-configurations
    app: AppSettings = AppSettings()
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    security: SecuritySettings = SecuritySettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    ai: AISettings = AISettings()
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.app.is_production
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.app.is_development


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Convenience exports
settings = get_settings() 