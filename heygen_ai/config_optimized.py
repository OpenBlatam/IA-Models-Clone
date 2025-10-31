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

import os
import sys
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from pydantic import BaseSettings, validator, Field
from pydantic.types import SecretStr
import logging
from functools import lru_cache
import json
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Optimized Configuration Management for HeyGen AI API
Enhanced settings with performance optimizations and better environment handling.
"""


logger = logging.getLogger(__name__)

# =============================================================================
# Environment Detection
# =============================================================================

def detect_environment() -> str:
    """Detect the current environment."""
    env = os.getenv("ENVIRONMENT", "").lower()
    if env in ["production", "prod"]:
        return "production"
    elif env in ["development", "dev"]:
        return "development"
    elif env in ["testing", "test"]:
        return "testing"
    elif env in ["staging", "stage"]:
        return "staging"
    else:
        # Auto-detect based on common patterns
        if os.getenv("PYTHON_ENV"):
            return os.getenv("PYTHON_ENV").lower()
        elif os.getenv("FLASK_ENV"):
            return os.getenv("FLASK_ENV").lower()
        elif os.getenv("NODE_ENV"):
            return os.getenv("NODE_ENV").lower()
        else:
            return "development"

# =============================================================================
# Base Settings
# =============================================================================

class BaseOptimizedSettings(BaseSettings):
    """Base settings with performance optimizations."""
    
    # Application settings
    app_name: str = "HeyGen AI Optimized API"
    app_version: str = "2.0.0"
    debug: bool = False
    environment: str = Field(default_factory=detect_environment)
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = Field(default=1, ge=1, le=32)
    reload: bool = False
    
    # Performance settings
    optimization_level: str = "standard"  # basic, standard, aggressive, custom
    enable_profiling: bool = False
    enable_metrics: bool = True
    enable_tracing: bool = False
    
    # Connection pooling
    max_database_connections: int = Field(default=50, ge=10, le=200)
    max_redis_connections: int = Field(default=50, ge=10, le=200)
    pool_timeout: int = Field(default=30, ge=5, le=60)
    pool_recycle: int = Field(default=3600, ge=300, le=7200)
    
    # Caching settings
    cache_enabled: bool = True
    cache_ttl: int = Field(default=300, ge=60, le=3600)
    memory_cache_size: int = Field(default=1000, ge=100, le=10000)
    redis_cache_enabled: bool = True
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = Field(default=100, ge=10, le=10000)
    rate_limit_window: int = Field(default=3600, ge=60, le=86400)
    
    # Database settings
    database_url: str = "sqlite+aiosqlite:///./heygen_ai.db"
    database_pool_size: int = Field(default=20, ge=5, le=100)
    database_max_overflow: int = Field(default=30, ge=5, le=100)
    database_echo: bool = False
    database_pool_pre_ping: bool = True
    
    # Redis settings
    redis_url: Optional[str] = None
    redis_max_connections: int = Field(default=50, ge=10, le=200)
    redis_health_check_interval: int = Field(default=30, ge=10, le=120)
    
    # Security settings
    secret_key: SecretStr = SecretStr("your-secret-key-change-in-production")
    api_key_header: str = "X-API-Key"
    jwt_secret: SecretStr = SecretStr("your-jwt-secret-change-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = Field(default=3600, ge=300, le=86400)
    
    # CORS settings
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    enable_request_logging: bool = True
    enable_sql_logging: bool = False
    
    # Monitoring settings
    metrics_port: int = Field(default=9090, ge=1024, le=65535)
    health_check_interval: int = Field(default=60, ge=30, le=300)
    slow_request_threshold_ms: float = Field(default=1000.0, ge=100.0, le=10000.0)
    
    # Model settings
    transformer_model_size: str = "medium"  # small, medium, large
    diffusion_pipeline: str = "stable-diffusion-v1-5"
    max_script_length: int = Field(default=1000, ge=100, le=10000)
    max_video_duration: int = Field(default=300, ge=30, le=1800)
    
    # Processing settings
    default_quality: str = "medium"
    max_concurrent_videos: int = Field(default=10, ge=1, le=100)
    video_output_dir: str = "outputs/videos"
    temp_dir: str = "temp"
    
    # External services
    openrouter_api_key: Optional[SecretStr] = None
    openai_api_key: Optional[SecretStr] = None
    huggingface_token: Optional[SecretStr] = None
    
    # Storage settings
    storage_type: str = "local"  # local, s3, gcs
    s3_bucket: Optional[str] = None
    s3_access_key: Optional[SecretStr] = None
    s3_secret_key: Optional[SecretStr] = None
    s3_region: Optional[str] = None
    
    # Email settings
    smtp_host: Optional[str] = None
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: Optional[str] = None
    smtp_password: Optional[SecretStr] = None
    smtp_use_tls: bool = True
    
    # Notification settings
    enable_notifications: bool = False
    webhook_url: Optional[str] = None
    
    # Auto-scaling settings
    auto_scaling_enabled: bool = True
    scaling_threshold: float = Field(default=0.8, ge=0.5, le=0.95)
    scaling_cooldown: int = Field(default=300, ge=60, le=1800)
    
    # Memory management
    gc_threshold: int = Field(default=1000, ge=100, le=10000)
    memory_limit_mb: Optional[int] = Field(default=None, ge=100, le=10000)
    
    # Async settings
    max_concurrent_requests: int = Field(default=1000, ge=100, le=10000)
    max_requests_per_worker: int = Field(default=10000, ge=1000, le=100000)
    keep_alive_timeout: int = Field(default=30, ge=5, le=60)
    graceful_shutdown_timeout: int = Field(default=30, ge=5, le=120)
    
    # Validation
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v) -> Any:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('cors_allow_methods', pre=True)
    def parse_cors_methods(cls, v) -> Any:
        if isinstance(v, str):
            return [method.strip() for method in v.split(',')]
        return v
    
    @validator('cors_allow_headers', pre=True)
    def parse_cors_headers(cls, v) -> Any:
        if isinstance(v, str):
            return [header.strip() for header in v.split(',')]
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v) -> bool:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()
    
    @validator('transformer_model_size')
    def validate_model_size(cls, v) -> bool:
        valid_sizes = ['small', 'medium', 'large']
        if v.lower() not in valid_sizes:
            raise ValueError(f'transformer_model_size must be one of {valid_sizes}')
        return v.lower()
    
    @validator('default_quality')
    def validate_quality(cls, v) -> bool:
        valid_qualities = ['low', 'medium', 'high']
        if v.lower() not in valid_qualities:
            raise ValueError(f'default_quality must be one of {valid_qualities}')
        return v.lower()
    
    @validator('storage_type')
    def validate_storage_type(cls, v) -> bool:
        valid_types = ['local', 's3', 'gcs']
        if v.lower() not in valid_types:
            raise ValueError(f'storage_type must be one of {valid_types}')
        return v.lower()
    
    @validator('optimization_level')
    def validate_optimization_level(cls, v) -> bool:
        valid_levels = ['basic', 'standard', 'aggressive', 'custom']
        if v.lower() not in valid_levels:
            raise ValueError(f'optimization_level must be one of {valid_levels}')
        return v.lower()
    
    @dataclass
class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True

# =============================================================================
# Environment-Specific Settings
# =============================================================================

class DevelopmentSettings(BaseOptimizedSettings):
    """Development environment settings with optimizations."""
    debug: bool = True
    reload: bool = True
    log_level: str = "DEBUG"
    enable_sql_logging: bool = True
    enable_profiling: bool = True
    
    # Development database
    database_url: str = "sqlite+aiosqlite:///./heygen_ai_dev.db"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Development security (less strict)
    cors_origins: List[str] = ["*"]
    
    # Development processing
    max_concurrent_videos: int = 5
    default_quality: str = "low"  # Faster for development
    max_script_length: int = 500
    
    # Development monitoring
    enable_metrics: bool = True
    enable_tracing: bool = True
    slow_request_threshold_ms: float = 500.0
    
    # Development caching
    memory_cache_size: int = 500
    cache_ttl: int = 60  # Shorter for development
    
    # Development auto-scaling
    auto_scaling_enabled: bool = False  # Disable for development

class ProductionSettings(BaseOptimizedSettings):
    """Production environment settings with optimizations."""
    debug: bool = False
    reload: bool = False
    log_level: str = "INFO"
    enable_sql_logging: bool = False
    enable_profiling: bool = False
    
    # Production security
    cors_origins: List[str] = ["https://yourdomain.com"]
    
    # Production processing
    max_concurrent_videos: int = 50
    default_quality: str = "medium"
    max_script_length: int = 2000
    
    # Production monitoring
    enable_metrics: bool = True
    enable_tracing: bool = False
    slow_request_threshold_ms: float = 2000.0
    
    # Production caching
    memory_cache_size: int = 2000
    cache_ttl: int = 600  # Longer for production
    
    # Production auto-scaling
    auto_scaling_enabled: bool = True
    scaling_threshold: float = 0.85
    scaling_cooldown: int = 600
    
    # Production connection pooling
    max_database_connections: int = 100
    max_redis_connections: int = 100
    database_pool_size: int = 50
    database_max_overflow: int = 50
    
    # Production storage
    storage_type: str = "s3"
    
    # Production rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600

class StagingSettings(BaseOptimizedSettings):
    """Staging environment settings with optimizations."""
    debug: bool = False
    reload: bool = False
    log_level: str = "INFO"
    enable_sql_logging: bool = False
    enable_profiling: bool = True  # Enable for staging testing
    
    # Staging security
    cors_origins: List[str] = ["https://staging.yourdomain.com"]
    
    # Staging processing
    max_concurrent_videos: int = 20
    default_quality: str = "medium"
    
    # Staging monitoring
    enable_metrics: bool = True
    enable_tracing: bool = True
    slow_request_threshold_ms: float = 1500.0
    
    # Staging caching
    memory_cache_size: int = 1000
    cache_ttl: int = 300
    
    # Staging auto-scaling
    auto_scaling_enabled: bool = True
    scaling_threshold: float = 0.8
    scaling_cooldown: int = 300

class TestingSettings(BaseOptimizedSettings):
    """Testing environment settings with optimizations."""
    debug: bool = True
    database_url: str = "sqlite+aiosqlite:///./test.db"
    
    # Testing security
    secret_key: SecretStr = SecretStr("test-secret-key")
    jwt_secret: SecretStr = SecretStr("test-jwt-secret")
    
    # Testing processing
    max_concurrent_videos: int = 1
    default_quality: str = "low"
    
    # Testing monitoring
    enable_metrics: bool = False
    enable_notifications: bool = False
    
    # Testing caching
    cache_enabled: bool = False
    redis_cache_enabled: bool = False
    
    # Testing auto-scaling
    auto_scaling_enabled: bool = False

# =============================================================================
# Settings Factory
# =============================================================================

@lru_cache()
def get_settings() -> BaseOptimizedSettings:
    """Get settings based on environment with caching."""
    environment = detect_environment()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "staging":
        return StagingSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()

# =============================================================================
# Configuration Utilities
# =============================================================================

def create_directories(settings: BaseOptimizedSettings):
    """Create necessary directories."""
    directories = [
        settings.video_output_dir,
        settings.temp_dir,
        "logs",
        "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def validate_settings(settings: BaseOptimizedSettings) -> List[str]:
    """Validate settings and return any issues."""
    issues = []
    
    # Check required settings for production
    if settings.environment == "production":
        if settings.secret_key.get_secret_value() == "your-secret-key-change-in-production":
            issues.append("Secret key must be changed in production")
        
        if settings.jwt_secret.get_secret_value() == "your-jwt-secret-change-in-production":
            issues.append("JWT secret must be changed in production")
        
        if settings.cors_origins == ["*"]:
            issues.append("CORS origins should be restricted in production")
        
        if settings.storage_type == "s3" and not settings.s3_bucket:
            issues.append("S3 bucket must be configured for S3 storage")
    
    # Check Redis configuration
    if settings.redis_cache_enabled and not settings.redis_url:
        issues.append("Redis URL must be configured when Redis caching is enabled")
    
    # Check external API keys
    if not settings.openrouter_api_key and not settings.openai_api_key:
        issues.append("At least one AI API key must be configured")
    
    return issues

def log_settings(settings: BaseOptimizedSettings):
    """Log current settings (excluding secrets)."""
    settings_dict = settings.dict()
    
    # Remove sensitive information
    sensitive_keys = ['secret_key', 'jwt_secret', 'openrouter_api_key', 'openai_api_key', 
                     'huggingface_token', 's3_access_key', 's3_secret_key', 'smtp_password']
    
    for key in sensitive_keys:
        if key in settings_dict:
            settings_dict[key] = "***HIDDEN***"
    
    logger.info("Application settings loaded", settings=settings_dict)

def setup_configuration():
    """Setup and validate configuration."""
    try:
        # Get settings
        settings = get_settings()
        
        # Create directories
        create_directories(settings)
        
        # Validate settings
        issues = validate_settings(settings)
        if issues:
            logger.warning("Configuration issues found", issues=issues)
            for issue in issues:
                logger.warning(f"Configuration issue: {issue}")
        
        # Log settings
        log_settings(settings)
        
        return settings
        
    except Exception as e:
        logger.error("Failed to setup configuration", error=str(e))
        raise

def get_environment_config(environment: str) -> Dict[str, Any]:
    """Get configuration for specific environment."""
    if environment == "production":
        return ProductionSettings().dict()
    elif environment == "staging":
        return StagingSettings().dict()
    elif environment == "testing":
        return TestingSettings().dict()
    else:
        return DevelopmentSettings().dict()

def is_development() -> bool:
    """Check if running in development mode."""
    return detect_environment() == "development"

def is_production() -> bool:
    """Check if running in production mode."""
    return detect_environment() == "production"

def is_testing() -> bool:
    """Check if running in testing mode."""
    return detect_environment() == "testing"

def is_staging() -> bool:
    """Check if running in staging mode."""
    return detect_environment() == "staging"

# =============================================================================
# Configuration Getters
# =============================================================================

def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    settings = get_settings()
    return {
        "url": settings.database_url,
        "pool_size": settings.database_pool_size,
        "max_overflow": settings.database_max_overflow,
        "pool_timeout": settings.pool_timeout,
        "pool_recycle": settings.pool_recycle,
        "echo": settings.database_echo,
        "pool_pre_ping": settings.database_pool_pre_ping
    }

def get_security_config() -> Dict[str, Any]:
    """Get security configuration."""
    settings = get_settings()
    return {
        "secret_key": settings.secret_key.get_secret_value(),
        "api_key_header": settings.api_key_header,
        "jwt_secret": settings.jwt_secret.get_secret_value(),
        "jwt_algorithm": settings.jwt_algorithm,
        "jwt_expiration": settings.jwt_expiration
    }

def get_cors_config() -> Dict[str, Any]:
    """Get CORS configuration."""
    settings = get_settings()
    return {
        "origins": settings.cors_origins,
        "allow_credentials": settings.cors_allow_credentials,
        "allow_methods": settings.cors_allow_methods,
        "allow_headers": settings.cors_allow_headers
    }

def get_rate_limit_config() -> Dict[str, Any]:
    """Get rate limiting configuration."""
    settings = get_settings()
    return {
        "enabled": settings.rate_limit_enabled,
        "requests": settings.rate_limit_requests,
        "window": settings.rate_limit_window
    }

def get_cache_config() -> Dict[str, Any]:
    """Get caching configuration."""
    settings = get_settings()
    return {
        "enabled": settings.cache_enabled,
        "ttl": settings.cache_ttl,
        "memory_size": settings.memory_cache_size,
        "redis_enabled": settings.redis_cache_enabled,
        "redis_url": settings.redis_url
    }

def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration."""
    settings = get_settings()
    return {
        "optimization_level": settings.optimization_level,
        "max_concurrent_requests": settings.max_concurrent_requests,
        "max_requests_per_worker": settings.max_requests_per_worker,
        "keep_alive_timeout": settings.keep_alive_timeout,
        "graceful_shutdown_timeout": settings.graceful_shutdown_timeout,
        "slow_request_threshold_ms": settings.slow_request_threshold_ms,
        "auto_scaling_enabled": settings.auto_scaling_enabled,
        "scaling_threshold": settings.scaling_threshold,
        "scaling_cooldown": settings.scaling_cooldown
    } 