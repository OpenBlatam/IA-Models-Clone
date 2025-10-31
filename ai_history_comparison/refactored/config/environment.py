"""
Environment Configuration
========================

This module handles environment-specific configuration and utilities.
"""

import os
from enum import Enum
from typing import Optional


class Environment(str, Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


def get_environment() -> Environment:
    """Get current environment from environment variable"""
    env_str = os.getenv("ENVIRONMENT", "development").lower()
    
    try:
        return Environment(env_str)
    except ValueError:
        # Default to development if invalid environment
        return Environment.DEVELOPMENT


def is_development() -> bool:
    """Check if running in development environment"""
    return get_environment() == Environment.DEVELOPMENT


def is_production() -> bool:
    """Check if running in production environment"""
    return get_environment() == Environment.PRODUCTION


def is_staging() -> bool:
    """Check if running in staging environment"""
    return get_environment() == Environment.STAGING


def is_testing() -> bool:
    """Check if running in testing environment"""
    return get_environment() == Environment.TESTING


def get_debug_mode() -> bool:
    """Get debug mode based on environment"""
    debug_env = os.getenv("DEBUG")
    if debug_env is not None:
        return debug_env.lower() in ("true", "1", "yes", "on")
    
    # Default based on environment
    return is_development()


def get_database_url() -> str:
    """Get database URL based on environment"""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url
    
    # Environment-specific defaults
    if is_production():
        return "postgresql://user:password@localhost/ai_history"
    elif is_testing():
        return "sqlite:///:memory:"
    else:
        return "sqlite:///./ai_history.db"


def get_log_level() -> str:
    """Get log level based on environment"""
    log_level = os.getenv("LOG_LEVEL")
    if log_level:
        return log_level.upper()
    
    # Environment-specific defaults
    if is_development():
        return "DEBUG"
    elif is_production():
        return "WARNING"
    else:
        return "INFO"


def get_cors_origins() -> list:
    """Get CORS origins based on environment"""
    cors_origins = os.getenv("CORS_ORIGINS")
    if cors_origins:
        return [origin.strip() for origin in cors_origins.split(",")]
    
    # Environment-specific defaults
    if is_development():
        return ["*"]
    elif is_production():
        return ["https://yourdomain.com"]
    else:
        return ["http://localhost:3000", "http://localhost:8000"]


def get_trusted_hosts() -> list:
    """Get trusted hosts based on environment"""
    trusted_hosts = os.getenv("TRUSTED_HOSTS")
    if trusted_hosts:
        return [host.strip() for host in trusted_hosts.split(",")]
    
    # Environment-specific defaults
    if is_development():
        return ["*"]
    elif is_production():
        return ["yourdomain.com", "*.yourdomain.com"]
    else:
        return ["localhost", "127.0.0.1"]


def get_api_key() -> Optional[str]:
    """Get API key from environment"""
    return os.getenv("API_KEY")


def get_secret_key() -> str:
    """Get secret key from environment"""
    secret_key = os.getenv("SECRET_KEY")
    if secret_key:
        return secret_key
    
    # Default for development (should be overridden in production)
    if is_development():
        return "development-secret-key-change-in-production"
    else:
        raise ValueError("SECRET_KEY environment variable is required in production")


def get_redis_url() -> Optional[str]:
    """Get Redis URL from environment"""
    return os.getenv("REDIS_URL")


def get_smtp_config() -> dict:
    """Get SMTP configuration from environment"""
    return {
        "host": os.getenv("SMTP_HOST"),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "username": os.getenv("SMTP_USERNAME"),
        "password": os.getenv("SMTP_PASSWORD"),
        "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true"
    }


def get_webhook_url() -> Optional[str]:
    """Get webhook URL from environment"""
    return os.getenv("WEBHOOK_URL")


def get_environment_info() -> dict:
    """Get comprehensive environment information"""
    return {
        "environment": get_environment().value,
        "debug": get_debug_mode(),
        "database_url": get_database_url(),
        "log_level": get_log_level(),
        "cors_origins": get_cors_origins(),
        "trusted_hosts": get_trusted_hosts(),
        "has_api_key": bool(get_api_key()),
        "has_secret_key": bool(get_secret_key()),
        "has_redis": bool(get_redis_url()),
        "has_smtp": bool(get_smtp_config()["host"]),
        "has_webhook": bool(get_webhook_url())
    }




