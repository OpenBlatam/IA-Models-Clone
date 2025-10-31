from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from typing import Optional, List, Dict, Any
from functools import lru_cache
from pydantic import BaseSettings, Field, validator
from pydantic.networks import RedisDsn, PostgresDsn
import os
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Application Settings - Centralized Configuration
===============================================

Environment-based configuration using Pydantic Settings v2
with validation, type safety, and environment variable support.
"""



class Environment(str, Enum):
    """Application environment types."""
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


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    # PostgreSQL
    postgres_server: str = Field(default="localhost", env="POSTGRES_SERVER")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="password", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="products", env="POSTGRES_DB")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    
    # Connection settings
    pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")
    
    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_server}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def async_database_url(self) -> str:
        """Get async PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_server}:{self.postgres_port}/{self.postgres_db}"


class RedisSettings(BaseSettings):
    """Redis configuration."""
    
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_ssl: bool = Field(default=False, env="REDIS_SSL")
    
    # Connection settings
    redis_max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    redis_timeout: int = Field(default=5, env="REDIS_TIMEOUT")
    redis_retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")
    
    # Cache TTL settings
    default_cache_ttl: int = Field(default=3600, env="DEFAULT_CACHE_TTL")  # 1 hour
    search_cache_ttl: int = Field(default=300, env="SEARCH_CACHE_TTL")    # 5 minutes
    analytics_cache_ttl: int = Field(default=600, env="ANALYTICS_CACHE_TTL")  # 10 minutes
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        protocol = "rediss" if self.redis_ssl else "redis"
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"{protocol}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


class SecuritySettings(BaseSettings):
    """Security configuration."""
    
    # API Keys
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    api_keys: List[str] = Field(default_factory=list, env="API_KEYS")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=1000, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    rate_limit_storage: str = Field(default="redis", env="RATE_LIMIT_STORAGE")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    
    # Request limits
    max_request_size: int = Field(default=10 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 10MB
    max_bulk_operations: int = Field(default=100, env="MAX_BULK_OPERATIONS")
    
    # JWT Settings (for future auth)
    jwt_secret_key: str = Field(default="your-secret-key-change-in-production", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=30, env="JWT_EXPIRE_MINUTES")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Metrics
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Health checks
    health_check_timeout: int = Field(default=10, env="HEALTH_CHECK_TIMEOUT")
    
    # Tracing
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    
    # Performance monitoring
    slow_query_threshold: float = Field(default=1.0, env="SLOW_QUERY_THRESHOLD")
    enable_profiling: bool = Field(default=False, env="ENABLE_PROFILING")


class APISettings(BaseSettings):
    """API-specific configuration."""
    
    # Basic info
    title: str = Field(default="Enhanced Product API", env="API_TITLE")
    description: str = Field(default="Modular product management API", env="API_DESCRIPTION")
    version: str = Field(default="3.0.0", env="API_VERSION")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    reload: bool = Field(default=False, env="API_RELOAD")
    workers: int = Field(default=1, env="API_WORKERS")
    
    # Documentation
    docs_url: str = Field(default="/docs", env="DOCS_URL")
    redoc_url: str = Field(default="/redoc", env="REDOC_URL")
    openapi_url: str = Field(default="/openapi.json", env="OPENAPI_URL")
    
    # Features
    enable_compression: bool = Field(default=True, env="ENABLE_COMPRESSION")
    compression_minimum_size: int = Field(default=1000, env="COMPRESSION_MIN_SIZE")
    
    # Pagination
    default_page_size: int = Field(default=20, env="DEFAULT_PAGE_SIZE")
    max_page_size: int = Field(default=100, env="MAX_PAGE_SIZE")
    
    # Search
    search_timeout: float = Field(default=5.0, env="SEARCH_TIMEOUT")
    max_search_results: int = Field(default=1000, env="MAX_SEARCH_RESULTS")


class AISettings(BaseSettings):
    """AI and ML configuration."""
    
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=500, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    
    # Text processing
    enable_ai_descriptions: bool = Field(default=False, env="ENABLE_AI_DESCRIPTIONS")
    ai_description_timeout: float = Field(default=10.0, env="AI_DESCRIPTION_TIMEOUT")
    ai_batch_size: int = Field(default=10, env="AI_BATCH_SIZE")
    
    # Content moderation
    enable_content_moderation: bool = Field(default=False, env="ENABLE_CONTENT_MODERATION")
    
    # Embeddings
    enable_embeddings: bool = Field(default=False, env="ENABLE_EMBEDDINGS")
    embeddings_model: str = Field(default="text-embedding-ada-002", env="EMBEDDINGS_MODEL")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    
    # Sub-configurations
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    security: SecuritySettings = SecuritySettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    api: APISettings = APISettings()
    ai: AISettings = AISettings()
    
    @dataclass
class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator('environment', pre=True)
    def validate_environment(cls, v) -> bool:
        if isinstance(v, str):
            return v.lower()
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing."""
        return self.environment == Environment.TESTING or self.testing
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        base_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s",
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json" if self.monitoring.log_format == "json" else "detailed",
                    "level": self.monitoring.log_level.value,
                },
            },
            "root": {
                "level": self.monitoring.log_level.value,
                "handlers": ["console"],
            },
            "loggers": {
                "uvicorn": {"level": "INFO"},
                "uvicorn.error": {"level": "INFO"},
                "uvicorn.access": {"level": "INFO"},
                "sqlalchemy.engine": {"level": "WARNING"},
                "redis": {"level": "WARNING"},
            }
        }
        
        # Add file handler if specified
        if self.monitoring.log_file:
            base_config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": self.monitoring.log_file,
                "formatter": "json" if self.monitoring.log_format == "json" else "detailed",
                "level": self.monitoring.log_level.value,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            }
            base_config["root"]["handlers"].append("file")
        
        return base_config


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Environment-specific configurations
def get_database_url(settings: Optional[Settings] = None) -> str:
    """Get database URL for current environment."""
    if settings is None:
        settings = get_settings()
    
    if settings.is_testing:
        return settings.database.database_url.replace(
            settings.database.postgres_db, 
            f"{settings.database.postgres_db}_test"
        )
    
    return settings.database.database_url


def get_redis_url(settings: Optional[Settings] = None) -> str:
    """Get Redis URL for current environment."""
    if settings is None:
        settings = get_settings()
    
    if settings.is_testing:
        # Use different Redis DB for testing
        test_db = settings.redis.redis_db + 1
        return settings.redis.redis_url.replace(f"/{settings.redis.redis_db}", f"/{test_db}")
    
    return settings.redis.redis_url


# Export main settings instance
settings = get_settings() 