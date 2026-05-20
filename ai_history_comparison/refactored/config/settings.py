"""
Application Settings
===================

This module defines the application settings using Pydantic for validation
and environment variable support.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from enum import Enum
import os


class Environment(str, Enum):
    """Application environment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    url: str = Field(default="sqlite:///./ai_history.db", env="DATABASE_URL")
    pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")
    echo: bool = Field(default=False, env="DB_ECHO")
    echo_pool: bool = Field(default=False, env="DB_ECHO_POOL")
    
    class Config:
        env_prefix = "DB_"


class APISettings(BaseSettings):
    """API configuration"""
    title: str = Field(default="AI History Comparison System", env="API_TITLE")
    description: str = Field(
        default="Comprehensive AI content analysis and comparison system",
        env="API_DESCRIPTION"
    )
    version: str = Field(default="2.0.0", env="API_VERSION")
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    cors_origins: List[str] = Field(default=["*"], env="API_CORS_ORIGINS")
    trusted_hosts: List[str] = Field(default=["*"], env="API_TRUSTED_HOSTS")
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("trusted_hosts", pre=True)
    def parse_trusted_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    class Config:
        env_prefix = "API_"


class SecuritySettings(BaseSettings):
    """Security configuration"""
    api_key: Optional[str] = Field(default=None, env="SECURITY_API_KEY")
    secret_key: str = Field(default="your-secret-key-here", env="SECURITY_SECRET_KEY")
    algorithm: str = Field(default="HS256", env="SECURITY_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="SECURITY_ACCESS_TOKEN_EXPIRE_MINUTES")
    enable_authentication: bool = Field(default=False, env="SECURITY_ENABLE_AUTH")
    enable_rate_limiting: bool = Field(default=True, env="SECURITY_ENABLE_RATE_LIMITING")
    requests_per_minute: int = Field(default=100, env="SECURITY_REQUESTS_PER_MINUTE")
    
    class Config:
        env_prefix = "SECURITY_"


class LoggingSettings(BaseSettings):
    """Logging configuration"""
    level: str = Field(default="INFO", env="LOGGING_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOGGING_FORMAT"
    )
    file_path: Optional[str] = Field(default=None, env="LOGGING_FILE_PATH")
    max_file_size: int = Field(default=10485760, env="LOGGING_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(default=5, env="LOGGING_BACKUP_COUNT")
    enable_console: bool = Field(default=True, env="LOGGING_ENABLE_CONSOLE")
    enable_file: bool = Field(default=False, env="LOGGING_ENABLE_FILE")
    
    class Config:
        env_prefix = "LOGGING_"


class CacheSettings(BaseSettings):
    """Cache configuration"""
    enable_cache: bool = Field(default=True, env="CACHE_ENABLE")
    cache_ttl: int = Field(default=300, env="CACHE_TTL")  # 5 minutes
    cache_size: int = Field(default=1000, env="CACHE_SIZE")
    redis_url: Optional[str] = Field(default=None, env="CACHE_REDIS_URL")
    
    class Config:
        env_prefix = "CACHE_"


class AnalysisSettings(BaseSettings):
    """Analysis configuration"""
    default_quality_threshold: float = Field(default=0.7, env="ANALYSIS_QUALITY_THRESHOLD")
    trend_analysis_window_days: int = Field(default=30, env="ANALYSIS_TREND_WINDOW_DAYS")
    comparison_minimum_samples: int = Field(default=10, env="ANALYSIS_COMPARISON_MIN_SAMPLES")
    anomaly_detection_sigma: float = Field(default=2.0, env="ANALYSIS_ANOMALY_SIGMA")
    forecast_days: int = Field(default=7, env="ANALYSIS_FORECAST_DAYS")
    confidence_threshold: float = Field(default=0.7, env="ANALYSIS_CONFIDENCE_THRESHOLD")
    
    class Config:
        env_prefix = "ANALYSIS_"


class NotificationSettings(BaseSettings):
    """Notification configuration"""
    enable_notifications: bool = Field(default=False, env="NOTIFICATION_ENABLE")
    email_enabled: bool = Field(default=False, env="NOTIFICATION_EMAIL_ENABLED")
    smtp_host: Optional[str] = Field(default=None, env="NOTIFICATION_SMTP_HOST")
    smtp_port: int = Field(default=587, env="NOTIFICATION_SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, env="NOTIFICATION_SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, env="NOTIFICATION_SMTP_PASSWORD")
    webhook_enabled: bool = Field(default=False, env="NOTIFICATION_WEBHOOK_ENABLED")
    webhook_url: Optional[str] = Field(default=None, env="NOTIFICATION_WEBHOOK_URL")
    
    class Config:
        env_prefix = "NOTIFICATION_"


class Settings(BaseSettings):
    """Main application settings"""
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: APISettings = Field(default_factory=APISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    notification: NotificationSettings = Field(default_factory=NotificationSettings)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator("debug", pre=True)
    def validate_debug(cls, v, values):
        if "environment" in values:
            return values["environment"] == Environment.DEVELOPMENT
        return v
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.environment == Environment.TESTING
    
    def get_database_url(self) -> str:
        """Get database URL with environment-specific defaults"""
        if self.database.url == "sqlite:///./ai_history.db" and self.is_production():
            # Use PostgreSQL in production
            return "postgresql://user:password@localhost/ai_history"
        return self.database.url
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins based on environment"""
        if self.is_development():
            return ["*"]
        elif self.is_production():
            return ["https://yourdomain.com", "https://www.yourdomain.com"]
        else:
            return self.api.cors_origins
    
    def get_trusted_hosts(self) -> List[str]:
        """Get trusted hosts based on environment"""
        if self.is_development():
            return ["*"]
        elif self.is_production():
            return ["yourdomain.com", "*.yourdomain.com"]
        else:
            return self.api.trusted_hosts
    
    def get_log_level(self) -> str:
        """Get log level based on environment"""
        if self.is_development():
            return "DEBUG"
        elif self.is_production():
            return "WARNING"
        else:
            return self.logging.level
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "database": self.database.dict(),
            "api": self.api.dict(),
            "security": self.security.dict(),
            "logging": self.logging.dict(),
            "cache": self.cache.dict(),
            "analysis": self.analysis.dict(),
            "notification": self.notification.dict()
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings():
    """Reload settings from environment"""
    global _settings
    _settings = Settings()
    return _settings




