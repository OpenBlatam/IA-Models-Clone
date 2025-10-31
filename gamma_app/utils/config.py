"""
Gamma App - Configuration Management
Centralized configuration for the application
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = "Gamma App"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    algorithm: str = "HS256"
    
    # Database
    database_url: str = Field("sqlite:///gamma_app.db", env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    # AI Services
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    openai_model: str = "gpt-4"
    anthropic_model: str = "claude-3-sonnet-20240229"
    
    # File Storage
    upload_dir: str = "uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: list = ["pdf", "pptx", "docx", "html", "png", "jpg", "jpeg"]
    
    # CORS
    cors_origins: list = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Email
    smtp_host: Optional[str] = Field(None, env="SMTP_HOST")
    smtp_port: int = 587
    smtp_username: Optional[str] = Field(None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(None, env="SMTP_PASSWORD")
    smtp_use_tls: bool = True
    
    # WebSocket
    websocket_ping_interval: int = 20
    websocket_ping_timeout: int = 10
    
    # Content Generation
    max_content_length: int = 10000
    default_content_style: str = "professional"
    default_output_format: str = "html"
    
    # Collaboration
    max_collaboration_sessions: int = 100
    session_timeout: int = 3600  # 1 hour
    max_participants_per_session: int = 10
    
    # Analytics
    analytics_retention_days: int = 90
    enable_analytics: bool = True
    
    # Cache
    cache_ttl: int = 3600  # 1 hour
    enable_cache: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get application settings"""
    return Settings()

def get_database_url() -> str:
    """Get database URL"""
    settings = get_settings()
    return settings.database_url

def get_redis_url() -> str:
    """Get Redis URL"""
    settings = get_settings()
    return settings.redis_url

def get_ai_config() -> Dict[str, Any]:
    """Get AI configuration"""
    settings = get_settings()
    return {
        "openai_api_key": settings.openai_api_key,
        "anthropic_api_key": settings.anthropic_api_key,
        "openai_model": settings.openai_model,
        "anthropic_model": settings.anthropic_model
    }

def get_security_config() -> Dict[str, Any]:
    """Get security configuration"""
    settings = get_settings()
    return {
        "secret_key": settings.secret_key,
        "access_token_expire_minutes": settings.access_token_expire_minutes,
        "refresh_token_expire_days": settings.refresh_token_expire_days,
        "algorithm": settings.algorithm
    }

def get_cors_config() -> Dict[str, Any]:
    """Get CORS configuration"""
    settings = get_settings()
    return {
        "origins": settings.cors_origins,
        "allow_credentials": settings.cors_allow_credentials,
        "allow_methods": settings.cors_allow_methods,
        "allow_headers": settings.cors_allow_headers
    }

def get_file_config() -> Dict[str, Any]:
    """Get file configuration"""
    settings = get_settings()
    return {
        "upload_dir": settings.upload_dir,
        "max_file_size": settings.max_file_size,
        "allowed_file_types": settings.allowed_file_types
    }

def get_websocket_config() -> Dict[str, Any]:
    """Get WebSocket configuration"""
    settings = get_settings()
    return {
        "ping_interval": settings.websocket_ping_interval,
        "ping_timeout": settings.websocket_ping_timeout
    }

def get_collaboration_config() -> Dict[str, Any]:
    """Get collaboration configuration"""
    settings = get_settings()
    return {
        "max_sessions": settings.max_collaboration_sessions,
        "session_timeout": settings.session_timeout,
        "max_participants": settings.max_participants_per_session
    }

def get_analytics_config() -> Dict[str, Any]:
    """Get analytics configuration"""
    settings = get_settings()
    return {
        "retention_days": settings.analytics_retention_days,
        "enabled": settings.enable_analytics
    }

def get_cache_config() -> Dict[str, Any]:
    """Get cache configuration"""
    settings = get_settings()
    return {
        "ttl": settings.cache_ttl,
        "enabled": settings.enable_cache
    }

def get_email_config() -> Dict[str, Any]:
    """Get email configuration"""
    settings = get_settings()
    return {
        "smtp_host": settings.smtp_host,
        "smtp_port": settings.smtp_port,
        "smtp_username": settings.smtp_username,
        "smtp_password": settings.smtp_password,
        "smtp_use_tls": settings.smtp_use_tls
    }

def get_monitoring_config() -> Dict[str, Any]:
    """Get monitoring configuration"""
    settings = get_settings()
    return {
        "enable_metrics": settings.enable_metrics,
        "metrics_port": settings.metrics_port
    }

def get_rate_limit_config() -> Dict[str, Any]:
    """Get rate limiting configuration"""
    settings = get_settings()
    return {
        "requests": settings.rate_limit_requests,
        "window": settings.rate_limit_window
    }

def get_content_config() -> Dict[str, Any]:
    """Get content generation configuration"""
    settings = get_settings()
    return {
        "max_length": settings.max_content_length,
        "default_style": settings.default_content_style,
        "default_format": settings.default_output_format
    }

def is_development() -> bool:
    """Check if running in development mode"""
    settings = get_settings()
    return settings.environment == "development"

def is_production() -> bool:
    """Check if running in production mode"""
    settings = get_settings()
    return settings.environment == "production"

def is_debug() -> bool:
    """Check if debug mode is enabled"""
    settings = get_settings()
    return settings.debug

def get_log_config() -> Dict[str, Any]:
    """Get logging configuration"""
    settings = get_settings()
    return {
        "level": settings.log_level,
        "format": settings.log_format
    }

def validate_config() -> bool:
    """Validate configuration"""
    try:
        settings = get_settings()
        
        # Check required settings
        if not settings.secret_key:
            raise ValueError("SECRET_KEY is required")
        
        # Check AI API keys (at least one should be provided)
        if not settings.openai_api_key and not settings.anthropic_api_key:
            raise ValueError("At least one AI API key (OpenAI or Anthropic) is required")
        
        # Check file upload directory
        if not os.path.exists(settings.upload_dir):
            os.makedirs(settings.upload_dir, exist_ok=True)
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

def get_environment_variables() -> Dict[str, str]:
    """Get all environment variables"""
    return dict(os.environ)

def get_config_summary() -> Dict[str, Any]:
    """Get configuration summary (without sensitive data)"""
    settings = get_settings()
    
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
        "api_host": settings.api_host,
        "api_port": settings.api_port,
        "database_url": settings.database_url.split("://")[0] + "://***",  # Hide credentials
        "redis_url": settings.redis_url.split("://")[0] + "://***",  # Hide credentials
        "openai_configured": bool(settings.openai_api_key),
        "anthropic_configured": bool(settings.anthropic_api_key),
        "max_file_size": settings.max_file_size,
        "cors_origins": settings.cors_origins,
        "rate_limit_requests": settings.rate_limit_requests,
        "log_level": settings.log_level,
        "enable_metrics": settings.enable_metrics,
        "enable_analytics": settings.enable_analytics,
        "enable_cache": settings.enable_cache
    }



























