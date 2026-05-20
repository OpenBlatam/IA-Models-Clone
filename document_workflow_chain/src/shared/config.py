"""
Configuration Management
========================

Advanced configuration management with environment variables,
validation, and type safety.
"""

from __future__ import annotations
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
from pathlib import Path

from pydantic import BaseSettings, Field, validator


logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """Database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"


class CacheBackend(str, Enum):
    """Cache backends"""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"


class AIProvider(str, Enum):
    """AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: DatabaseType = DatabaseType.POSTGRESQL
    host: str = "localhost"
    port: int = 5432
    name: str = "workflow_chain"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    ssl_mode: str = "prefer"
    
    @property
    def url(self) -> str:
        """Get database URL"""
        if self.type == DatabaseType.SQLITE:
            return f"sqlite:///{self.name}.db"
        elif self.type == DatabaseType.POSTGRESQL:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.name}"
        elif self.type == DatabaseType.MYSQL:
            return f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.name}"
        else:
            raise ValueError(f"Unsupported database type: {self.type}")


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    max_connections: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    
    @property
    def url(self) -> str:
        """Get Redis URL"""
        scheme = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class AIConfig:
    """AI service configuration"""
    provider: AIProvider = AIProvider.OPENAI
    api_key: str = ""
    base_url: Optional[str] = None
    model: str = "gpt-4"
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3
    rate_limit_per_minute: int = 60


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    jwt_refresh_expire_days: int = 7
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = False
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_allow_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class CacheConfig:
    """Cache configuration"""
    backend: CacheBackend = CacheBackend.MEMORY
    default_ttl: int = 300
    max_size: int = 1000
    compression: bool = True
    serialization: str = "json"
    key_prefix: str = "workflow:"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    enabled: bool = True
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    window_size: int = 60


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    metrics_enabled: bool = True
    health_check_enabled: bool = True
    profiling_enabled: bool = False
    tracing_enabled: bool = False
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    log_file: Optional[str] = None
    max_log_size: int = 100  # MB
    backup_count: int = 5


@dataclass
class NotificationConfig:
    """Notification configuration"""
    email_enabled: bool = True
    sms_enabled: bool = False
    push_enabled: bool = True
    webhook_enabled: bool = False
    slack_enabled: bool = False
    teams_enabled: bool = False
    discord_enabled: bool = False
    
    # Email settings
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    
    # Slack settings
    slack_webhook_url: str = ""
    slack_bot_token: str = ""
    
    # Webhook settings
    webhook_url: str = ""
    webhook_secret: str = ""


@dataclass
class AnalyticsConfig:
    """Analytics configuration"""
    enabled: bool = True
    batch_size: int = 100
    flush_interval: int = 60
    retention_days: int = 90
    anonymize_data: bool = False
    real_time_enabled: bool = True
    storage_type: str = "memory"


@dataclass
class AuditConfig:
    """Audit configuration"""
    enabled: bool = True
    log_level: str = "info"
    retention_days: int = 365
    batch_size: int = 100
    flush_interval: int = 30
    mask_sensitive_data: bool = True
    gdpr_compliant: bool = True
    sox_compliant: bool = False
    hipaa_compliant: bool = False


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = "Document Workflow Chain v3.0"
    app_version: str = "3.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Database
    database_type: DatabaseType = DatabaseType.POSTGRESQL
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "workflow_chain"
    database_username: str = "postgres"
    database_password: str = ""
    database_pool_size: int = 10
    database_echo: bool = False
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # AI
    ai_provider: AIProvider = AIProvider.OPENAI
    ai_api_key: str = ""
    ai_model: str = "gpt-4"
    ai_max_tokens: int = 4000
    ai_temperature: float = 0.7
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    
    # Cache
    cache_backend: CacheBackend = CacheBackend.MEMORY
    cache_default_ttl: int = 300
    cache_max_size: int = 1000
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 100
    rate_limit_per_hour: int = 1000
    
    # Monitoring
    monitoring_enabled: bool = True
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    
    # Notifications
    notification_email_enabled: bool = True
    notification_slack_enabled: bool = False
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    
    # Analytics
    analytics_enabled: bool = True
    analytics_retention_days: int = 90
    
    # Audit
    audit_enabled: bool = True
    audit_retention_days: int = 365
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator('environment')
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator('database_type')
    def validate_database_type(cls, v):
        if isinstance(v, str):
            return DatabaseType(v.lower())
        return v
    
    @validator('ai_provider')
    def validate_ai_provider(cls, v):
        if isinstance(v, str):
            return AIProvider(v.lower())
        return v
    
    @validator('cache_backend')
    def validate_cache_backend(cls, v):
        if isinstance(v, str):
            return CacheBackend(v.lower())
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return DatabaseConfig(
            type=self.database_type,
            host=self.database_host,
            port=self.database_port,
            name=self.database_name,
            username=self.database_username,
            password=self.database_password,
            pool_size=self.database_pool_size,
            echo=self.database_echo
        )
    
    def get_redis_config(self) -> RedisConfig:
        """Get Redis configuration"""
        return RedisConfig(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password
        )
    
    def get_ai_config(self) -> AIConfig:
        """Get AI configuration"""
        return AIConfig(
            provider=self.ai_provider,
            api_key=self.ai_api_key,
            model=self.ai_model,
            max_tokens=self.ai_max_tokens,
            temperature=self.ai_temperature
        )
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return SecurityConfig(
            secret_key=self.secret_key,
            jwt_algorithm=self.jwt_algorithm,
            jwt_expire_minutes=self.jwt_expire_minutes
        )
    
    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration"""
        return CacheConfig(
            backend=self.cache_backend,
            default_ttl=self.cache_default_ttl,
            max_size=self.cache_max_size
        )
    
    def get_rate_limit_config(self) -> RateLimitConfig:
        """Get rate limit configuration"""
        return RateLimitConfig(
            enabled=self.rate_limit_enabled,
            requests_per_minute=self.rate_limit_per_minute,
            requests_per_hour=self.rate_limit_per_hour
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return MonitoringConfig(
            enabled=self.monitoring_enabled,
            log_level=self.log_level,
            log_format=self.log_format
        )
    
    def get_notification_config(self) -> NotificationConfig:
        """Get notification configuration"""
        return NotificationConfig(
            email_enabled=self.notification_email_enabled,
            slack_enabled=self.notification_slack_enabled,
            smtp_host=self.smtp_host,
            smtp_port=self.smtp_port,
            smtp_username=self.smtp_username,
            smtp_password=self.smtp_password
        )
    
    def get_analytics_config(self) -> AnalyticsConfig:
        """Get analytics configuration"""
        return AnalyticsConfig(
            enabled=self.analytics_enabled,
            retention_days=self.analytics_retention_days
        )
    
    def get_audit_config(self) -> AuditConfig:
        """Get audit configuration"""
        return AuditConfig(
            enabled=self.audit_enabled,
            retention_days=self.audit_retention_days
        )
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.environment == Environment.TESTING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment.value,
            "debug": self.debug,
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "database": {
                "type": self.database_type.value,
                "host": self.database_host,
                "port": self.database_port,
                "name": self.database_name,
                "username": self.database_username,
                "pool_size": self.database_pool_size,
                "echo": self.database_echo
            },
            "redis": {
                "host": self.redis_host,
                "port": self.redis_port,
                "db": self.redis_db
            },
            "ai": {
                "provider": self.ai_provider.value,
                "model": self.ai_model,
                "max_tokens": self.ai_max_tokens,
                "temperature": self.ai_temperature
            },
            "security": {
                "jwt_algorithm": self.jwt_algorithm,
                "jwt_expire_minutes": self.jwt_expire_minutes
            },
            "cache": {
                "backend": self.cache_backend.value,
                "default_ttl": self.cache_default_ttl,
                "max_size": self.cache_max_size
            },
            "rate_limit": {
                "enabled": self.rate_limit_enabled,
                "requests_per_minute": self.rate_limit_per_minute,
                "requests_per_hour": self.rate_limit_per_hour
            },
            "monitoring": {
                "enabled": self.monitoring_enabled,
                "log_level": self.log_level.value,
                "log_format": self.log_format
            },
            "notifications": {
                "email_enabled": self.notification_email_enabled,
                "slack_enabled": self.notification_slack_enabled
            },
            "analytics": {
                "enabled": self.analytics_enabled,
                "retention_days": self.analytics_retention_days
            },
            "audit": {
                "enabled": self.audit_enabled,
                "retention_days": self.audit_retention_days
            }
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def load_settings_from_file(file_path: str) -> Settings:
    """Load settings from file"""
    try:
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        # Update environment variables
        for key, value in config_data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    env_key = f"{key.upper()}_{sub_key.upper()}"
                    os.environ[env_key] = str(sub_value)
            else:
                os.environ[key.upper()] = str(value)
        
        return Settings()
        
    except Exception as e:
        logger.error(f"Failed to load settings from file {file_path}: {e}")
        return Settings()


def validate_settings(settings: Settings) -> List[str]:
    """Validate settings and return list of issues"""
    issues = []
    
    # Validate required fields
    if not settings.secret_key or settings.secret_key == "your-secret-key-change-in-production":
        issues.append("Secret key must be set and changed from default")
    
    if settings.is_production() and settings.debug:
        issues.append("Debug mode should not be enabled in production")
    
    if settings.database_password == "" and settings.is_production():
        issues.append("Database password must be set in production")
    
    if settings.ai_api_key == "" and settings.ai_provider != AIProvider.OPENAI:
        issues.append("AI API key must be set")
    
    # Validate port
    if not (1 <= settings.port <= 65535):
        issues.append("Port must be between 1 and 65535")
    
    # Validate pool size
    if settings.database_pool_size < 1:
        issues.append("Database pool size must be at least 1")
    
    return issues


def setup_logging(settings: Settings) -> None:
    """Setup logging configuration"""
    import logging.config
    
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "json": {
                "format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level.value,
                "formatter": "json" if settings.log_format == "json" else "default",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": settings.log_level.value,
            "handlers": ["console"],
        },
    }
    
    # Add file handler if specified
    if settings.log_file:
        log_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.log_level.value,
            "formatter": "json" if settings.log_format == "json" else "default",
            "filename": settings.log_file,
            "maxBytes": settings.max_log_size * 1024 * 1024,  # Convert MB to bytes
            "backupCount": settings.backup_count,
        }
        log_config["root"]["handlers"].append("file")
    
    logging.config.dictConfig(log_config)
    logger.info(f"Logging configured with level {settings.log_level.value}")


# Initialize settings and logging
settings = get_settings()
setup_logging(settings)

# Validate settings
validation_issues = validate_settings(settings)
if validation_issues:
    logger.warning("Settings validation issues found:")
    for issue in validation_issues:
        logger.warning(f"  - {issue}")
    
    if settings.is_production():
        logger.error("Production settings validation failed")
        raise ValueError("Invalid production settings")




