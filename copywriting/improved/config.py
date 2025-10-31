"""
Configuration Management
=======================

Centralized configuration using Pydantic Settings.
"""

import os
from typing import Optional, List
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    url: str = Field(default="sqlite:///./copywriting.db", description="Database URL")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max overflow connections")
    echo: bool = Field(default=False, description="Enable SQL echo")


class RedisSettings(BaseSettings):
    """Redis configuration"""
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    max_connections: int = Field(default=10, ge=1, le=100, description="Max Redis connections")
    socket_timeout: int = Field(default=5, ge=1, le=60, description="Socket timeout in seconds")
    socket_connect_timeout: int = Field(default=5, ge=1, le=60, description="Connection timeout in seconds")


class APISettings(BaseSettings):
    """API configuration"""
    model_config = SettingsConfigDict(env_prefix="API_")
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, ge=1, le=65535, description="API port")
    workers: int = Field(default=1, ge=1, le=32, description="Number of workers")
    reload: bool = Field(default=False, description="Enable auto-reload")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    cors_methods: List[str] = Field(default=["*"], description="CORS allowed methods")
    cors_headers: List[str] = Field(default=["*"], description="CORS allowed headers")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, ge=1, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window in seconds")
    
    @field_validator('cors_origins', 'cors_methods', 'cors_headers')
    @classmethod
    def validate_cors_lists(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("CORS lists cannot be empty")
        return v


class SecuritySettings(BaseSettings):
    """Security configuration"""
    model_config = SettingsConfigDict(env_prefix="SECURITY_")
    
    secret_key: str = Field(default="your-secret-key-here", min_length=32, description="Secret key for JWT")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, ge=1, le=1440, description="Access token expiry")
    refresh_token_expire_days: int = Field(default=7, ge=1, le=30, description="Refresh token expiry")
    
    # API Key settings
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    api_keys: List[str] = Field(default_factory=list, description="Valid API keys")
    
    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v


class LoggingSettings(BaseSettings):
    """Logging configuration"""
    model_config = SettingsConfigDict(env_prefix="LOG_")
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10485760, ge=1024, description="Max log file size in bytes")
    backup_count: int = Field(default=5, ge=1, le=20, description="Number of backup files")
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


class CacheSettings(BaseSettings):
    """Cache configuration"""
    model_config = SettingsConfigDict(env_prefix="CACHE_")
    
    enabled: bool = Field(default=True, description="Enable caching")
    default_ttl: int = Field(default=300, ge=1, le=3600, description="Default TTL in seconds")
    max_size: int = Field(default=1000, ge=1, le=10000, description="Max cache size")
    
    # Cache keys
    copywriting_ttl: int = Field(default=600, ge=1, le=3600, description="Copywriting cache TTL")
    feedback_ttl: int = Field(default=1800, ge=1, le=3600, description="Feedback cache TTL")


class MonitoringSettings(BaseSettings):
    """Monitoring configuration"""
    model_config = SettingsConfigDict(env_prefix="MONITORING_")
    
    enabled: bool = Field(default=True, description="Enable monitoring")
    metrics_endpoint: str = Field(default="/metrics", description="Metrics endpoint")
    health_endpoint: str = Field(default="/health", description="Health check endpoint")
    
    # Prometheus settings
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=9090, ge=1, le=65535, description="Prometheus port")
    
    # Performance monitoring
    track_response_times: bool = Field(default=True, description="Track response times")
    track_memory_usage: bool = Field(default=True, description="Track memory usage")
    track_cpu_usage: bool = Field(default=True, description="Track CPU usage")


class Settings(BaseSettings):
    """Main application settings"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    version: str = Field(default="2.0.0", description="Application version")
    
    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid_envs = ["development", "staging", "production", "testing"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


@lru_cache()
def get_database_settings() -> DatabaseSettings:
    """Get cached database settings"""
    return get_settings().database


@lru_cache()
def get_redis_settings() -> RedisSettings:
    """Get cached Redis settings"""
    return get_settings().redis


@lru_cache()
def get_api_settings() -> APISettings:
    """Get cached API settings"""
    return get_settings().api


@lru_cache()
def get_security_settings() -> SecuritySettings:
    """Get cached security settings"""
    return get_settings().security


@lru_cache()
def get_logging_settings() -> LoggingSettings:
    """Get cached logging settings"""
    return get_settings().logging


@lru_cache()
def get_cache_settings() -> CacheSettings:
    """Get cached cache settings"""
    return get_settings().cache


@lru_cache()
def get_monitoring_settings() -> MonitoringSettings:
    """Get cached monitoring settings"""
    return get_settings().monitoring































