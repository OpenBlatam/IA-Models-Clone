"""
Configuration management for Facebook Posts API
Following functional programming principles and Pydantic best practices
"""

from typing import Optional, Dict, Any, List, Union
from pydantic import BaseSettings, Field, validator, root_validator
import os
from pathlib import Path
import secrets


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # API Configuration
    api_title: str = Field("Ultimate Facebook Posts API", env="API_TITLE")
    api_version: str = Field("4.0.0", env="API_VERSION")
    api_description: str = Field("AI-powered Facebook post generation system", env="API_DESCRIPTION")
    debug: bool = Field(False, env="DEBUG")
    
    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT", ge=1, le=65535)
    workers: int = Field(1, env="WORKERS", ge=1, le=32)
    max_request_size: int = Field(10485760, env="MAX_REQUEST_SIZE", ge=1024)  # 10MB
    
    # Database Configuration
    database_url: str = Field("sqlite:///./facebook_posts.db", env="DATABASE_URL")
    database_pool_size: int = Field(10, env="DATABASE_POOL_SIZE", ge=1, le=100)
    database_max_overflow: int = Field(20, env="DATABASE_MAX_OVERFLOW", ge=0, le=100)
    
    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    cache_default_ttl: int = Field(3600, env="CACHE_DEFAULT_TTL", ge=60)
    cache_max_connections: int = Field(10, env="CACHE_MAX_CONNECTIONS", ge=1, le=100)
    
    # AI Configuration
    ai_api_key: str = Field("", env="AI_API_KEY")
    ai_model: str = Field("gpt-3.5-turbo", env="AI_MODEL")
    ai_max_tokens: int = Field(2000, env="AI_MAX_TOKENS", ge=100, le=4000)
    ai_temperature: float = Field(0.7, env="AI_TEMPERATURE", ge=0.0, le=2.0)
    ai_timeout: int = Field(30, env="AI_TIMEOUT", ge=5, le=300)
    
    # Analytics Configuration
    analytics_api_key: Optional[str] = Field(None, env="ANALYTICS_API_KEY")
    analytics_endpoint: Optional[str] = Field(None, env="ANALYTICS_ENDPOINT")
    analytics_timeout: int = Field(10, env="ANALYTICS_TIMEOUT", ge=5, le=60)
    
    # Rate Limiting
    rate_limit_requests: int = Field(1000, env="RATE_LIMIT_REQUESTS", ge=1)
    rate_limit_window: int = Field(3600, env="RATE_LIMIT_WINDOW", ge=1)
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    
    # Performance Configuration
    max_concurrent_requests: int = Field(1000, env="MAX_CONCURRENT_REQUESTS", ge=1, le=10000)
    request_timeout: int = Field(30, env="REQUEST_TIMEOUT", ge=5, le=300)
    background_task_timeout: int = Field(300, env="BACKGROUND_TASK_TIMEOUT", ge=30, le=1800)
    
    # Security Configuration
    cors_origins: str = Field("*", env="CORS_ORIGINS")
    cors_methods: str = Field("GET,POST,PUT,DELETE,OPTIONS", env="CORS_METHODS")
    cors_headers: str = Field("*", env="CORS_HEADERS")
    api_key: str = Field("", env="API_KEY")
    secret_key: str = Field("", env="SECRET_KEY")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_expiration: int = Field(3600, env="JWT_EXPIRATION", ge=300, le=86400)  # 5 min to 24 hours
    
    # Engine Configuration
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    enable_health_checks: bool = Field(True, env="ENABLE_HEALTH_CHECKS")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of: {", ".join(valid_levels)}')
        return v.upper()
    
    @validator('cors_origins')
    def validate_cors_origins(cls, v):
        if not v:
            return ["*"]
        return [origin.strip() for origin in v.split(",")]
    
    @validator('cors_methods')
    def validate_cors_methods(cls, v):
        if not v:
            return ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        return [method.strip() for method in v.split(",")]
    
    @validator('cors_headers')
    def validate_cors_headers(cls, v):
        if not v:
            return ["*"]
        return [header.strip() for header in v.split(",")]
    
    @validator('database_url')
    def validate_database_url(cls, v):
        if not v or not v.startswith(('postgresql://', 'mysql://', 'sqlite://')):
            raise ValueError('database_url must be a valid database URL')
        return v
    
    @validator('redis_url')
    def validate_redis_url(cls, v):
        if not v or not v.startswith(('redis://', 'rediss://')):
            raise ValueError('redis_url must be a valid Redis URL')
        return v
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if not v:
            # Generate a secure random key if not provided
            return secrets.token_urlsafe(32)
        if len(v) < 16:
            raise ValueError('secret_key must be at least 16 characters long')
        return v
    
    @validator('jwt_algorithm')
    def validate_jwt_algorithm(cls, v):
        valid_algorithms = ['HS256', 'HS384', 'HS512', 'RS256', 'RS384', 'RS512']
        if v not in valid_algorithms:
            raise ValueError(f'jwt_algorithm must be one of: {", ".join(valid_algorithms)}')
        return v
    
    @root_validator
    def validate_security_settings(cls, values):
        """Validate security-related settings"""
        debug = values.get('debug', False)
        api_key = values.get('api_key', '')
        secret_key = values.get('secret_key', '')
        
        # In production, require API key
        if not debug and not api_key:
            raise ValueError('api_key is required in production mode')
        
        # Ensure secret key is set
        if not secret_key:
            values['secret_key'] = secrets.token_urlsafe(32)
        
        return values
    
    @root_validator
    def validate_performance_settings(cls, values):
        """Validate performance-related settings"""
        max_concurrent = values.get('max_concurrent_requests', 1000)
        request_timeout = values.get('request_timeout', 30)
        background_timeout = values.get('background_task_timeout', 300)
        
        # Ensure background timeout is greater than request timeout
        if background_timeout <= request_timeout:
            raise ValueError('background_task_timeout must be greater than request_timeout')
        
        return values
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


# Pure functions for configuration management

def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment"""
    global _settings
    _settings = Settings()
    return _settings


def get_database_config() -> Dict[str, Any]:
    """Get database configuration"""
    settings = get_settings()
    return {
        "url": settings.database_url,
        "pool_size": settings.database_pool_size,
        "max_overflow": settings.database_max_overflow,
        "echo": settings.debug
    }


def get_redis_config() -> Dict[str, Any]:
    """Get Redis configuration"""
    settings = get_settings()
    return {
        "url": settings.redis_url,
        "max_connections": settings.cache_max_connections,
        "default_ttl": settings.cache_default_ttl
    }


def get_ai_config() -> Dict[str, Any]:
    """Get AI service configuration"""
    settings = get_settings()
    return {
        "api_key": settings.ai_api_key,
        "model": settings.ai_model,
        "max_tokens": settings.ai_max_tokens,
        "temperature": settings.ai_temperature,
        "timeout": settings.ai_timeout
    }


def get_analytics_config() -> Dict[str, Any]:
    """Get analytics configuration"""
    settings = get_settings()
    return {
        "api_key": settings.analytics_api_key,
        "endpoint": settings.analytics_endpoint,
        "timeout": settings.analytics_timeout
    }


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration"""
    settings = get_settings()
    config = {
        "level": settings.log_level,
        "format": settings.log_format,
        "handlers": ["console"]
    }
    
    if settings.log_file:
        config["handlers"].append("file")
        config["file"] = {
            "filename": settings.log_file,
            "max_bytes": 10485760,  # 10MB
            "backup_count": 5
        }
    
    return config


def get_cors_config() -> Dict[str, Any]:
    """Get CORS configuration"""
    settings = get_settings()
    return {
        "allow_origins": settings.cors_origins,
        "allow_methods": settings.cors_methods,
        "allow_headers": settings.cors_headers,
        "allow_credentials": True
    }


def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration"""
    settings = get_settings()
    return {
        "max_concurrent_requests": settings.max_concurrent_requests,
        "request_timeout": settings.request_timeout,
        "background_task_timeout": settings.background_task_timeout,
        "enable_caching": settings.enable_caching,
        "enable_metrics": settings.enable_metrics
    }


def get_security_config() -> Dict[str, Any]:
    """Get security configuration"""
    settings = get_settings()
    return {
        "api_key": settings.api_key,
        "secret_key": settings.secret_key,
        "jwt_algorithm": settings.jwt_algorithm,
        "jwt_expiration": settings.jwt_expiration,
        "rate_limit_requests": settings.rate_limit_requests,
        "rate_limit_window": settings.rate_limit_window,
        "cors_config": get_cors_config()
    }


def get_jwt_config() -> Dict[str, Any]:
    """Get JWT configuration"""
    settings = get_settings()
    return {
        "secret_key": settings.secret_key,
        "algorithm": settings.jwt_algorithm,
        "expiration": settings.jwt_expiration
    }


def validate_environment() -> bool:
    """Validate environment configuration"""
    try:
        settings = get_settings()
        
        # Check required settings
        required_settings = [
            "database_url", "redis_url"
        ]
        
        missing_settings = []
        for setting in required_settings:
            if not getattr(settings, setting, None):
                missing_settings.append(setting)
        
        if missing_settings:
            raise ValueError(f"Missing required settings: {', '.join(missing_settings)}")
        
        # Check file paths
        if settings.log_file:
            log_path = Path(settings.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        return True
        
    except Exception as e:
        print(f"Environment validation failed: {e}")
        return False


def is_development_mode() -> bool:
    """Check if running in development mode"""
    settings = get_settings()
    return settings.debug


def is_production_mode() -> bool:
    """Check if running in production mode"""
    return not is_development_mode()


def get_api_info() -> Dict[str, str]:
    """Get API information"""
    settings = get_settings()
    return {
        "title": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description
    }