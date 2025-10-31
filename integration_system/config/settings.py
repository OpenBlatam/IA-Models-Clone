"""
Settings Configuration
======================

Centralized configuration management for the integration system.
"""

import os
from typing import Dict, Any, Optional, List
from pydantic import BaseSettings, Field
from pathlib import Path
import yaml
import json

class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    app_name: str = Field("Blatam Academy Integration System", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    debug: bool = Field(False, env="DEBUG")
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Security settings
    secret_key: str = Field("your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = Field("HS256", env="ALGORITHM")
    
    # Database settings
    database_url: str = Field("sqlite:///./integration.db", env="DATABASE_URL")
    database_echo: bool = Field(False, env="DATABASE_ECHO")
    
    # Redis settings
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    redis_enabled: bool = Field(True, env="REDIS_ENABLED")
    
    # System endpoints
    content_redundancy_endpoint: str = Field("http://localhost:8001", env="CONTENT_REDUNDANCY_ENDPOINT")
    bul_endpoint: str = Field("http://localhost:8002", env="BUL_ENDPOINT")
    gamma_app_endpoint: str = Field("http://localhost:8003", env="GAMMA_APP_ENDPOINT")
    business_agents_endpoint: str = Field("http://localhost:8004", env="BUSINESS_AGENTS_ENDPOINT")
    export_ia_endpoint: str = Field("http://localhost:8005", env="EXPORT_IA_ENDPOINT")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(1000, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(3600, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Health check settings
    health_check_interval: int = Field(30, env="HEALTH_CHECK_INTERVAL")  # seconds
    health_check_timeout: int = Field(10, env="HEALTH_CHECK_TIMEOUT")  # seconds
    
    # Request timeout
    request_timeout: int = Field(30, env="REQUEST_TIMEOUT")  # seconds
    
    # File upload settings
    max_file_size: int = Field(50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    allowed_file_types: List[str] = Field(
        ["pdf", "docx", "txt", "md", "html", "json", "xml"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # CORS settings
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")
    cors_methods: List[str] = Field(["*"], env="CORS_METHODS")
    cors_headers: List[str] = Field(["*"], env="CORS_HEADERS")
    
    # Monitoring settings
    metrics_enabled: bool = Field(True, env="METRICS_ENABLED")
    metrics_interval: int = Field(60, env="METRICS_INTERVAL")  # seconds
    
    # Logging settings
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Cache settings
    cache_enabled: bool = Field(True, env="CACHE_ENABLED")
    cache_ttl: int = Field(300, env="CACHE_TTL")  # seconds
    
    # Integration settings
    integration_retry_attempts: int = Field(3, env="INTEGRATION_RETRY_ATTEMPTS")
    integration_retry_delay: int = Field(1, env="INTEGRATION_RETRY_DELAY")  # seconds
    
    # System-specific settings
    content_redundancy_settings: Dict[str, Any] = Field(default_factory=dict)
    bul_settings: Dict[str, Any] = Field(default_factory=dict)
    gamma_app_settings: Dict[str, Any] = Field(default_factory=dict)
    business_agents_settings: Dict[str, Any] = Field(default_factory=dict)
    export_ia_settings: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file."""
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_file.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")

def update_settings_from_config(config: Dict[str, Any]) -> None:
    """Update global settings from a configuration dictionary."""
    global _settings
    
    if _settings is None:
        _settings = Settings()
    
    # Update settings with configuration values
    for key, value in config.items():
        if hasattr(_settings, key):
            setattr(_settings, key, value)

def get_system_config(system_name: str) -> Dict[str, Any]:
    """Get configuration for a specific system."""
    settings = get_settings()
    
    system_configs = {
        "content_redundancy": settings.content_redundancy_settings,
        "bul": settings.bul_settings,
        "gamma_app": settings.gamma_app_settings,
        "business_agents": settings.business_agents_settings,
        "export_ia": settings.export_ia_settings
    }
    
    return system_configs.get(system_name, {})

def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    settings = get_settings()
    
    return {
        "url": settings.database_url,
        "echo": settings.database_echo,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_recycle": 3600
    }

def get_redis_config() -> Dict[str, Any]:
    """Get Redis configuration."""
    settings = get_settings()
    
    return {
        "url": settings.redis_url,
        "enabled": settings.redis_enabled,
        "decode_responses": True,
        "socket_connect_timeout": 5,
        "socket_timeout": 5,
        "retry_on_timeout": True,
        "health_check_interval": 30
    }

def get_cors_config() -> Dict[str, Any]:
    """Get CORS configuration."""
    settings = get_settings()
    
    return {
        "allow_origins": settings.cors_origins,
        "allow_methods": settings.cors_methods,
        "allow_headers": settings.cors_headers,
        "allow_credentials": True,
        "max_age": 3600
    }

def get_rate_limit_config() -> Dict[str, Any]:
    """Get rate limiting configuration."""
    settings = get_settings()
    
    return {
        "enabled": settings.rate_limit_enabled,
        "requests": settings.rate_limit_requests,
        "window": settings.rate_limit_window,
        "storage": "redis" if settings.redis_enabled else "memory"
    }

def get_health_check_config() -> Dict[str, Any]:
    """Get health check configuration."""
    settings = get_settings()
    
    return {
        "interval": settings.health_check_interval,
        "timeout": settings.health_check_timeout,
        "retry_attempts": 3,
        "retry_delay": 1
    }

def get_system_endpoints() -> Dict[str, str]:
    """Get all system endpoints."""
    settings = get_settings()
    
    return {
        "content_redundancy": settings.content_redundancy_endpoint,
        "bul": settings.bul_endpoint,
        "gamma_app": settings.gamma_app_endpoint,
        "business_agents": settings.business_agents_endpoint,
        "export_ia": settings.export_ia_endpoint
    }

def validate_config() -> List[str]:
    """Validate configuration and return any issues."""
    issues = []
    settings = get_settings()
    
    # Check required settings
    if not settings.secret_key or settings.secret_key == "your-secret-key-here":
        issues.append("SECRET_KEY must be set to a secure value")
    
    # Check endpoints
    endpoints = get_system_endpoints()
    for system, endpoint in endpoints.items():
        if not endpoint.startswith(("http://", "https://")):
            issues.append(f"Invalid endpoint for {system}: {endpoint}")
    
    # Check file size limits
    if settings.max_file_size <= 0:
        issues.append("MAX_FILE_SIZE must be greater than 0")
    
    # Check rate limiting
    if settings.rate_limit_requests <= 0:
        issues.append("RATE_LIMIT_REQUESTS must be greater than 0")
    
    if settings.rate_limit_window <= 0:
        issues.append("RATE_LIMIT_WINDOW must be greater than 0")
    
    return issues

def get_environment_info() -> Dict[str, Any]:
    """Get environment information."""
    settings = get_settings()
    
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "debug": settings.debug,
        "host": settings.host,
        "port": settings.port,
        "log_level": settings.log_level,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "python_version": os.sys.version,
        "platform": os.name
    }

