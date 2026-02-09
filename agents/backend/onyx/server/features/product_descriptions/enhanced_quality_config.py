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
import secrets
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urlparse
from pydantic import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Enhanced Quality Configuration Module
===================================

Enterprise-grade configuration with comprehensive validation,
type safety, and production-ready patterns.
"""


    BaseSettings, Field, validator, root_validator,
    PostgresDsn, RedisDsn, AnyHttpUrl, EmailStr
)


class Environment(str, Enum):
    """Application environment with validation."""
    LOCAL = "local"
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels with proper ordering."""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class DatabaseConfig(BaseSettings):
    """Database configuration with connection validation."""
    
    # Connection settings
    url: PostgresDsn = Field(
        default="postgresql+asyncpg://user:pass@localhost:5432/products",
        env="DATABASE_URL",
        description="PostgreSQL connection URL"
    )
    
    # Pool settings with validation
    pool_size: int = Field(
        default=10, ge=1, le=50, env="DB_POOL_SIZE",
        description="Connection pool size"
    )
    max_overflow: int = Field(
        default=20, ge=0, le=100, env="DB_MAX_OVERFLOW",
        description="Maximum pool overflow"
    )
    pool_timeout: int = Field(
        default=30, ge=5, le=300, env="DB_POOL_TIMEOUT",
        description="Pool connection timeout in seconds"
    )
    pool_recycle: int = Field(
        default=3600, ge=300, le=86400, env="DB_POOL_RECYCLE",
        description="Pool connection recycle time in seconds"
    )
    
    # Query settings
    echo: bool = Field(default=False, env="DB_ECHO")
    echo_pool: bool = Field(default=False, env="DB_ECHO_POOL")
    
    # Health and monitoring
    ping_timeout: float = Field(
        default=5.0, ge=1.0, le=30.0, env="DB_PING_TIMEOUT",
        description="Database ping timeout"
    )
    
    @validator('url')
    def validate_database_url(cls, v) -> bool:
        """Validate database URL format and scheme."""
        if not v:
            raise ValueError("Database URL cannot be empty")
        
        parsed = urlparse(str(v))
        if parsed.scheme not in ['postgresql', 'postgresql+asyncpg']:
            raise ValueError("Database must use PostgreSQL")
        
        return v
    
    @root_validator
    def validate_pool_settings(cls, values) -> bool:
        """Validate pool configuration coherence."""
        pool_size = values.get('pool_size', 10)
        max_overflow = values.get('max_overflow', 20)
        
        if max_overflow < pool_size:
            raise ValueError("max_overflow must be >= pool_size")
        
        total_connections = pool_size + max_overflow
        if total_connections > 100:
            raise ValueError("Total connections (pool_size + max_overflow) cannot exceed 100")
        
        return values


class RedisConfig(BaseSettings):
    """Redis configuration with cluster support."""
    
    # Connection
    url: RedisDsn = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL",
        description="Redis connection URL"
    )
    
    # Connection pooling
    max_connections: int = Field(
        default=20, ge=1, le=100, env="REDIS_MAX_CONNECTIONS",
        description="Maximum Redis connections"
    )
    timeout: float = Field(
        default=5.0, ge=1.0, le=60.0, env="REDIS_TIMEOUT",
        description="Connection timeout in seconds"
    )
    retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")
    
    # Cache TTL settings (in seconds)
    default_ttl: int = Field(
        default=3600, ge=60, le=86400, env="CACHE_DEFAULT_TTL",
        description="Default cache TTL (1 hour)"
    )
    search_ttl: int = Field(
        default=300, ge=30, le=3600, env="CACHE_SEARCH_TTL",
        description="Search results TTL (5 minutes)"
    )
    analytics_ttl: int = Field(
        default=600, ge=60, le=7200, env="CACHE_ANALYTICS_TTL",
        description="Analytics cache TTL (10 minutes)"
    )
    
    # Cluster settings
    cluster_enabled: bool = Field(default=False, env="REDIS_CLUSTER_ENABLED")
    cluster_nodes: List[str] = Field(default_factory=list, env="REDIS_CLUSTER_NODES")
    
    @validator('url')
    def validate_redis_url(cls, v) -> bool:
        """Validate Redis URL format."""
        if not v:
            raise ValueError("Redis URL cannot be empty")
        
        parsed = urlparse(str(v))
        if parsed.scheme not in ['redis', 'rediss']:
            raise ValueError("Redis URL must use redis:// or rediss:// scheme")
        
        return v


class SecurityConfig(BaseSettings):
    """Security configuration with enhanced validation."""
    
    # Secrets and keys
    secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        env="SECRET_KEY",
        min_length=32,
        description="Application secret key"
    )
    
    # API Security
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    api_keys: Set[str] = Field(default_factory=set, env="API_KEYS")
    
    # Rate limiting with environment-based defaults
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    requests_per_minute: int = Field(
        default=100, ge=10, le=10000, env="REQUESTS_PER_MINUTE"
    )
    requests_per_hour: int = Field(
        default=1000, ge=100, le=100000, env="REQUESTS_PER_HOUR"
    )
    burst_limit: int = Field(
        default=20, ge=5, le=100, env="RATE_LIMIT_BURST"
    )
    
    # CORS with validation
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    
    # Request limits
    max_request_size: int = Field(
        default=10_000_000, ge=1_000_000, le=100_000_000, env="MAX_REQUEST_SIZE",
        description="Maximum request size in bytes (10MB)"
    )
    max_bulk_operations: int = Field(
        default=100, ge=1, le=1000, env="MAX_BULK_OPERATIONS"
    )
    
    # JWT settings
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(
        default=30, ge=5, le=1440, env="JWT_EXPIRE_MINUTES"
    )
    
    # Security headers
    enable_security_headers: bool = Field(default=True, env="ENABLE_SECURITY_HEADERS")
    hsts_max_age: int = Field(default=31536000, env="HSTS_MAX_AGE")  # 1 year
    
    @validator('secret_key')
    def validate_secret_key(cls, v) -> bool:
        """Validate secret key strength."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v
    
    @validator('cors_origins')
    def validate_cors_origins(cls, v) -> bool:
        """Validate CORS origins format."""
        for origin in v:
            if origin != "*" and not origin.startswith(("http://", "https://")):
                raise ValueError(f"Invalid CORS origin format: {origin}")
        return v
    
    @root_validator
    def validate_rate_limits(cls, values) -> bool:
        """Validate rate limiting configuration."""
        per_minute = values.get('requests_per_minute', 100)
        per_hour = values.get('requests_per_hour', 1000)
        
        # Per minute rate shouldn't exceed hourly rate
        if per_minute * 60 > per_hour:
            raise ValueError("Hourly rate limit is inconsistent with per-minute rate")
        
        return values


class ObservabilityConfig(BaseSettings):
    """Observability and monitoring configuration."""
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(
        default="json", regex="^(json|text)$", env="LOG_FORMAT"
    )
    log_file: Optional[Path] = Field(default=None, env="LOG_FILE")
    log_rotation_size: str = Field(default="10MB", env="LOG_ROTATION_SIZE")
    log_retention_days: int = Field(
        default=30, ge=1, le=365, env="LOG_RETENTION_DAYS"
    )
    
    # Metrics and monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(
        default=9090, ge=1024, le=65535, env="METRICS_PORT"
    )
    metrics_path: str = Field(default="/metrics", env="METRICS_PATH")
    
    # Tracing
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    jaeger_endpoint: Optional[AnyHttpUrl] = Field(default=None, env="JAEGER_ENDPOINT")
    trace_sample_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, env="TRACE_SAMPLE_RATE"
    )
    
    # Error tracking
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    sentry_environment: Optional[str] = Field(default=None, env="SENTRY_ENVIRONMENT")
    sentry_sample_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, env="SENTRY_SAMPLE_RATE"
    )
    
    # Performance monitoring
    slow_query_threshold: float = Field(
        default=1.0, ge=0.1, le=60.0, env="SLOW_QUERY_THRESHOLD"
    )
    enable_profiling: bool = Field(default=False, env="ENABLE_PROFILING")
    profile_sample_rate: float = Field(
        default=0.01, ge=0.0, le=1.0, env="PROFILE_SAMPLE_RATE"
    )
    
    # Health checks
    health_check_timeout: float = Field(
        default=10.0, ge=1.0, le=60.0, env="HEALTH_CHECK_TIMEOUT"
    )
    
    @validator('log_file')
    def validate_log_file(cls, v) -> bool:
        """Validate log file path and ensure directory exists."""
        if v:
            log_path = Path(v)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        return v


class EnhancedAppConfig(BaseSettings):
    """Enhanced application configuration with comprehensive validation."""
    
    # Application metadata
    name: str = Field(
        default="Enhanced Product API",
        min_length=1, max_length=100,
        env="APP_NAME"
    )
    version: str = Field(
        default="2.1.0",
        regex=r"^\d+\.\d+\.\d+(-\w+)?$",
        env="APP_VERSION"
    )
    description: str = Field(
        default="Enterprise-grade product management API",
        max_length=500,
        env="APP_DESCRIPTION"
    )
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, ge=1024, le=65535, env="PORT")
    workers: int = Field(default=1, ge=1, le=32, env="WORKERS")
    reload: bool = Field(default=False, env="RELOAD")
    
    # API settings
    api_prefix: str = Field(
        default="/api/v2",
        regex=r"^/api/v\d+$",
        env="API_PREFIX"
    )
    docs_url: Optional[str] = Field(default="/docs", env="DOCS_URL")
    redoc_url: Optional[str] = Field(default="/redoc", env="REDOC_URL")
    openapi_url: Optional[str] = Field(default="/openapi.json", env="OPENAPI_URL")
    
    # Features
    enable_compression: bool = Field(default=True, env="ENABLE_COMPRESSION")
    compression_minimum_size: int = Field(
        default=1000, ge=100, le=10000, env="COMPRESSION_MIN_SIZE"
    )
    
    # Pagination
    default_page_size: int = Field(
        default=20, ge=1, le=100, env="DEFAULT_PAGE_SIZE"
    )
    max_page_size: int = Field(
        default=100, ge=10, le=1000, env="MAX_PAGE_SIZE"
    )
    
    # AI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(
        default=500, ge=50, le=4000, env="OPENAI_MAX_TOKENS"
    )
    openai_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, env="OPENAI_TEMPERATURE"
    )
    ai_timeout: float = Field(
        default=30.0, ge=5.0, le=120.0, env="AI_TIMEOUT"
    )
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    security: SecurityConfig = SecurityConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    
    @dataclass
class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True
        arbitrary_types_allowed = True
    
    @validator('environment', pre=True)
    def validate_environment(cls, v) -> bool:
        """Validate and normalize environment."""
        if isinstance(v, str):
            return v.lower()
        return v
    
    @validator('debug')
    def validate_debug_with_environment(cls, v, values) -> bool:
        """Debug should be disabled in production."""
        env = values.get('environment')
        if env == Environment.PRODUCTION and v:
            raise ValueError("Debug mode cannot be enabled in production")
        return v
    
    @root_validator
    def validate_configuration_coherence(cls, values) -> bool:
        """Validate overall configuration coherence."""
        environment = values.get('environment')
        debug = values.get('debug')
        
        # Production validations
        if environment == Environment.PRODUCTION:
            if debug:
                raise ValueError("Debug cannot be enabled in production")
            
            # Ensure secure defaults in production
            security_config = values.get('security', {})
            if hasattr(security_config, 'cors_origins'):
                if '*' in security_config.cors_origins:
                    raise ValueError("Wildcard CORS origins not allowed in production")
        
        return values
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment in [Environment.DEVELOPMENT, Environment.LOCAL]
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing."""
        return self.environment == Environment.TESTING
    
    @property
    def ai_enabled(self) -> bool:
        """Check if AI features are enabled."""
        return bool(self.openai_api_key)
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get structured logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d"
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json" if self.observability.log_format == "json" else "detailed",
                    "level": self.observability.log_level.value,
                    "stream": "ext://sys.stdout"
                }
            },
            "root": {
                "level": self.observability.log_level.value,
                "handlers": ["console"]
            },
            "loggers": {
                "uvicorn": {"level": "INFO", "propagate": False, "handlers": ["console"]},
                "uvicorn.error": {"level": "INFO", "propagate": False, "handlers": ["console"]},
                "uvicorn.access": {"level": "INFO", "propagate": False, "handlers": ["console"]},
                "sqlalchemy.engine": {"level": "WARNING", "propagate": False, "handlers": ["console"]},
                "redis": {"level": "WARNING", "propagate": False, "handlers": ["console"]},
                "httpx": {"level": "WARNING", "propagate": False, "handlers": ["console"]},
            }
        }


@lru_cache()
def get_enhanced_config() -> EnhancedAppConfig:
    """Get cached enhanced configuration."""
    return EnhancedAppConfig()


# Global configuration instance
config = get_enhanced_config()


# Configuration validation
def validate_config() -> bool:
    """Validate configuration on startup."""
    try:
        config = get_enhanced_config()
        
        # Additional runtime validations
        if config.is_production:
            required_prod_vars = [
                "SECRET_KEY", "DATABASE_URL", "REDIS_URL"
            ]
            missing_vars = [var for var in required_prod_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(f"Missing required production environment variables: {missing_vars}")
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False 