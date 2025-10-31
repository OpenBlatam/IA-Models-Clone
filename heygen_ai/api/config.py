from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
from typing import Optional, List
from pydantic import BaseSettings, validator
from pathlib import Path
import logging
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Configuration management for HeyGen AI API
Environment-based settings with validation.
"""


logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    app_name: str = "HeyGen AI Equivalent API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Database settings
    database_url: str = "sqlite+aiosqlite:///./heygen_ai.db"
    
    # Security settings
    secret_key: str = "your-secret-key-change-in-production"
    api_key_header: str = "X-API-Key"
    jwt_secret: str = "your-jwt-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # 1 hour
    
    # CORS settings
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # Model settings
    transformer_model_size: str = "medium"  # small, medium, large
    diffusion_pipeline: str = "stable-diffusion-v1-5"
    max_script_length: int = 1000
    max_video_duration: int = 300  # 5 minutes
    
    # Processing settings
    default_quality: str = "medium"
    max_concurrent_videos: int = 10
    video_output_dir: str = "outputs/videos"
    temp_dir: str = "temp"
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    
    # External services
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    
    # Storage settings
    storage_type: str = "local"  # local, s3, gcs
    s3_bucket: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_region: Optional[str] = None
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    redis_url: Optional[str] = None
    
    # Email settings
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    
    # Notification settings
    enable_notifications: bool = False
    webhook_url: Optional[str] = None
    
    # Development settings
    reload: bool = False
    log_sql: bool = False
    
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
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    @validator('transformer_model_size')
    def validate_model_size(cls, v) -> bool:
        valid_sizes = ['small', 'medium', 'large']
        if v not in valid_sizes:
            raise ValueError(f'Model size must be one of {valid_sizes}')
        return v
    
    @validator('default_quality')
    def validate_quality(cls, v) -> bool:
        valid_qualities = ['low', 'medium', 'high']
        if v not in valid_qualities:
            raise ValueError(f'Quality must be one of {valid_qualities}')
        return v
    
    @validator('storage_type')
    def validate_storage_type(cls, v) -> bool:
        valid_types = ['local', 's3', 'gcs']
        if v not in valid_types:
            raise ValueError(f'Storage type must be one of {valid_types}')
        return v
    
    @dataclass
class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class DevelopmentSettings(Settings):
    """Development environment settings."""
    debug: bool = True
    reload: bool = True
    log_sql: bool = True
    log_level: str = "DEBUG"
    
    # Development database
    database_url: str = "sqlite+aiosqlite:///./heygen_ai_dev.db"
    
    # Development security (less strict)
    cors_origins: List[str] = ["*"]
    
    # Development processing
    max_concurrent_videos: int = 5
    default_quality: str = "low"  # Faster for development


class ProductionSettings(Settings):
    """Production environment settings."""
    debug: bool = False
    reload: bool = False
    log_sql: bool = False
    log_level: str = "INFO"
    
    # Production security
    cors_origins: List[str] = ["https://yourdomain.com"]
    
    # Production processing
    max_concurrent_videos: int = 50
    default_quality: str = "medium"
    
    # Production monitoring
    enable_metrics: bool = True
    enable_notifications: bool = True
    
    # Production storage
    storage_type: str = "s3"
    
    # Production rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 1000


class TestingSettings(Settings):
    """Testing environment settings."""
    debug: bool = True
    database_url: str = "sqlite+aiosqlite:///./test.db"
    
    # Testing security
    secret_key: str = "test-secret-key"
    jwt_secret: str = "test-jwt-secret"
    
    # Testing processing
    max_concurrent_videos: int = 1
    default_quality: str = "low"
    
    # Testing monitoring
    enable_metrics: bool = False
    enable_notifications: bool = False


def get_settings() -> Settings:
    """Get settings based on environment."""
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Global settings instance
settings = get_settings()


def create_directories():
    """Create necessary directories."""
    directories = [
        settings.video_output_dir,
        settings.temp_dir,
        "logs",
        "uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def validate_settings():
    """Validate critical settings."""
    errors = []
    
    # Check required settings
    if not settings.secret_key or settings.secret_key == "your-secret-key-change-in-production":
        errors.append("Secret key must be set in production")
    
    if settings.storage_type == "s3":
        if not all([settings.s3_bucket, settings.s3_access_key, settings.s3_secret_key]):
            errors.append("S3 storage requires bucket, access key, and secret key")
    
    if settings.enable_notifications and not settings.webhook_url:
        errors.append("Notifications enabled but webhook URL not set")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")


def log_settings():
    """Log current settings (without sensitive data)."""
    safe_settings = {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "debug": settings.debug,
        "host": settings.host,
        "port": settings.port,
        "database_url": settings.database_url.split("://")[0] + "://***",  # Hide sensitive parts
        "transformer_model_size": settings.transformer_model_size,
        "diffusion_pipeline": settings.diffusion_pipeline,
        "max_concurrent_videos": settings.max_concurrent_videos,
        "default_quality": settings.default_quality,
        "storage_type": settings.storage_type,
        "rate_limit_enabled": settings.rate_limit_enabled,
        "cache_enabled": settings.cache_enabled,
        "enable_metrics": settings.enable_metrics,
        "log_level": settings.log_level
    }
    
    logger.info("Application settings:")
    for key, value in safe_settings.items():
        logger.info(f"  {key}: {value}")


# Configuration validation and setup
def setup_configuration():
    """Setup and validate configuration."""
    try:
        # Create directories
        create_directories()
        
        # Validate settings
        validate_settings()
        
        # Log settings
        log_settings()
        
        logger.info("✓ Configuration setup completed")
        
    except Exception as e:
        logger.error(f"Configuration setup failed: {e}")
        raise


# Environment-specific configurations
ENVIRONMENT_CONFIGS = {
    "development": {
        "database_url": "sqlite+aiosqlite:///./heygen_ai_dev.db",
        "log_level": "DEBUG",
        "debug": True,
        "reload": True
    },
    "production": {
        "database_url": "postgresql+asyncpg://user:pass@localhost/heygen_ai",
        "log_level": "INFO",
        "debug": False,
        "reload": False
    },
    "testing": {
        "database_url": "sqlite+aiosqlite:///./test.db",
        "log_level": "DEBUG",
        "debug": True,
        "reload": False
    }
}


def get_environment_config(environment: str) -> dict:
    """Get environment-specific configuration."""
    return ENVIRONMENT_CONFIGS.get(environment, ENVIRONMENT_CONFIGS["development"])


# Configuration utilities
def is_development() -> bool:
    """Check if running in development mode."""
    return settings.debug and settings.reload


def is_production() -> bool:
    """Check if running in production mode."""
    return not settings.debug and not settings.reload


def is_testing() -> bool:
    """Check if running in testing mode."""
    return settings.debug and not settings.reload


# Database configuration helpers
def get_database_config() -> dict:
    """Get database configuration."""
    return {
        "url": settings.database_url,
        "echo": settings.log_sql,
        "pool_pre_ping": True,
        "pool_recycle": 300
    }


# Security configuration helpers
def get_security_config() -> dict:
    """Get security configuration."""
    return {
        "secret_key": settings.secret_key,
        "jwt_secret": settings.jwt_secret,
        "jwt_algorithm": settings.jwt_algorithm,
        "jwt_expiration": settings.jwt_expiration
    }


# CORS configuration helpers
def get_cors_config() -> dict:
    """Get CORS configuration."""
    return {
        "allow_origins": settings.cors_origins,
        "allow_credentials": settings.cors_allow_credentials,
        "allow_methods": settings.cors_allow_methods,
        "allow_headers": settings.cors_allow_headers
    }


# Rate limiting configuration helpers
def get_rate_limit_config() -> dict:
    """Get rate limiting configuration."""
    return {
        "enabled": settings.rate_limit_enabled,
        "requests": settings.rate_limit_requests,
        "window": settings.rate_limit_window
    }


# Cache configuration helpers
def get_cache_config() -> dict:
    """Get cache configuration."""
    return {
        "enabled": settings.cache_enabled,
        "ttl": settings.cache_ttl,
        "redis_url": settings.redis_url
    } 