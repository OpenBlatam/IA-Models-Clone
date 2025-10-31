"""
Configuration Management for Improved Video-OpusClip API

Comprehensive configuration system with:
- Environment-based settings
- Type-safe configuration
- Validation and defaults
- Security best practices
- Performance optimization settings
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import os
from pathlib import Path
from pydantic import BaseSettings, Field, validator
from enum import Enum

# =============================================================================
# ENVIRONMENT TYPES
# =============================================================================

class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# =============================================================================
# BASE CONFIGURATION
# =============================================================================

class BaseConfig(BaseSettings):
    """Base configuration class with common settings."""
    
    # Application
    app_name: str = Field(default="video-opusclip-api", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    app_description: str = Field(
        default="Enhanced Video-OpusClip API with FastAPI best practices",
        description="Application description"
    )
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=4, description="Number of worker processes")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production", description="Secret key")
    jwt_secret_key: str = Field(default="your-jwt-secret-key-change-in-production", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_access_token_expire_minutes: int = Field(default=30, description="JWT token expiration")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    cors_allow_credentials: bool = Field(default=True, description="CORS allow credentials")
    cors_allow_methods: List[str] = Field(default=["GET", "POST"], description="CORS allowed methods")
    cors_allow_headers: List[str] = Field(default=["*"], description="CORS allowed headers")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

class DatabaseConfig(BaseConfig):
    """Database configuration settings."""
    
    # Database URL
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/video_api",
        description="Database connection URL"
    )
    
    # Connection Pool
    database_pool_size: int = Field(default=20, description="Database pool size")
    database_max_overflow: int = Field(default=30, description="Database max overflow")
    database_pool_timeout: int = Field(default=30, description="Database pool timeout")
    database_pool_recycle: int = Field(default=3600, description="Database pool recycle")
    
    # Connection Settings
    database_echo: bool = Field(default=False, description="Echo SQL queries")
    database_echo_pool: bool = Field(default=False, description="Echo pool events")
    
    @validator('database_url')
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format."""
        if not v.startswith(('postgresql://', 'postgresql+asyncpg://')):
            raise ValueError("Database URL must be a valid PostgreSQL URL")
        return v

# =============================================================================
# REDIS CONFIGURATION
# =============================================================================

class RedisConfig(BaseConfig):
    """Redis configuration settings."""
    
    # Redis Connection
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_ssl: bool = Field(default=False, description="Redis SSL connection")
    
    # Redis Pool
    redis_pool_size: int = Field(default=20, description="Redis connection pool size")
    redis_pool_timeout: int = Field(default=30, description="Redis pool timeout")
    redis_pool_retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    
    # Redis Settings
    redis_socket_keepalive: bool = Field(default=True, description="Redis socket keepalive")
    redis_socket_keepalive_options: Dict[str, Any] = Field(
        default={},
        description="Redis socket keepalive options"
    )
    
    @property
    def redis_url(self) -> str:
        """Get Redis URL."""
        protocol = "rediss" if self.redis_ssl else "redis"
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"{protocol}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

class CacheConfig(BaseConfig):
    """Cache configuration settings."""
    
    # Cache Settings
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_default_ttl: int = Field(default=3600, description="Default cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache size")
    
    # Fallback Cache
    cache_enable_fallback: bool = Field(default=True, description="Enable fallback cache")
    cache_fallback_max_size: int = Field(default=100, description="Fallback cache max size")
    
    # Cache Keys
    cache_key_prefix: str = Field(default="video_api:", description="Cache key prefix")
    cache_key_separator: str = Field(default=":", description="Cache key separator")
    
    # Cache Performance
    cache_compress: bool = Field(default=True, description="Compress cache data")
    cache_serializer: str = Field(default="json", description="Cache serializer")

# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================

class MonitoringConfig(BaseConfig):
    """Monitoring configuration settings."""
    
    # Performance Monitoring
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    performance_metrics_interval: int = Field(default=60, description="Metrics collection interval")
    performance_metrics_retention: int = Field(default=7, description="Metrics retention in days")
    
    # Health Checks
    enable_health_checks: bool = Field(default=True, description="Enable health checks")
    health_check_interval: int = Field(default=30, description="Health check interval")
    health_check_timeout: int = Field(default=10, description="Health check timeout")
    
    # Prometheus
    enable_prometheus: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")
    prometheus_endpoint: str = Field(default="/metrics", description="Prometheus endpoint")
    
    # Logging
    enable_structured_logging: bool = Field(default=True, description="Enable structured logging")
    log_request_id: bool = Field(default=True, description="Log request IDs")
    log_response_time: bool = Field(default=True, description="Log response times")

# =============================================================================
# PROCESSOR CONFIGURATION
# =============================================================================

class VideoProcessorConfig(BaseConfig):
    """Video processor configuration settings."""
    
    # Processing Settings
    max_workers: int = Field(default=8, description="Maximum worker threads")
    timeout: float = Field(default=300.0, description="Processing timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    
    # Quality Settings
    default_quality: str = Field(default="high", description="Default video quality")
    supported_qualities: List[str] = Field(
        default=["low", "medium", "high", "ultra"],
        description="Supported video qualities"
    )
    
    # Format Settings
    default_format: str = Field(default="mp4", description="Default video format")
    supported_formats: List[str] = Field(
        default=["mp4", "avi", "mov", "mkv"],
        description="Supported video formats"
    )
    
    # Resource Limits
    max_file_size_mb: int = Field(default=500, description="Maximum file size in MB")
    max_duration_seconds: int = Field(default=600, description="Maximum video duration")
    memory_limit_mb: int = Field(default=1024, description="Memory limit in MB")

class ViralProcessorConfig(BaseConfig):
    """Viral processor configuration settings."""
    
    # Variant Settings
    max_variants: int = Field(default=10, description="Maximum number of variants")
    min_viral_score: float = Field(default=0.3, description="Minimum viral score")
    
    # Platform Settings
    supported_platforms: List[str] = Field(
        default=["youtube", "tiktok", "instagram", "twitter", "linkedin"],
        description="Supported platforms"
    )
    
    # LangChain Integration
    enable_langchain: bool = Field(default=True, description="Enable LangChain integration")
    langchain_timeout: float = Field(default=300.0, description="LangChain timeout")
    
    # Optimization Settings
    enable_screen_division: bool = Field(default=True, description="Enable screen division")
    enable_transitions: bool = Field(default=True, description="Enable transitions")
    enable_effects: bool = Field(default=True, description="Enable effects")
    enable_animations: bool = Field(default=True, description="Enable animations")

class LangChainConfig(BaseConfig):
    """LangChain configuration settings."""
    
    # Model Settings
    model_name: str = Field(default="gpt-4", description="LLM model name")
    model_temperature: float = Field(default=0.7, description="Model temperature")
    model_max_tokens: int = Field(default=2000, description="Maximum tokens")
    
    # Analysis Settings
    enable_content_analysis: bool = Field(default=True, description="Enable content analysis")
    enable_engagement_analysis: bool = Field(default=True, description="Enable engagement analysis")
    enable_viral_analysis: bool = Field(default=True, description="Enable viral analysis")
    enable_title_optimization: bool = Field(default=True, description="Enable title optimization")
    enable_caption_optimization: bool = Field(default=True, description="Enable caption optimization")
    enable_timing_optimization: bool = Field(default=True, description="Enable timing optimization")
    
    # Processing Settings
    batch_size: int = Field(default=5, description="Batch size for processing")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    use_agents: bool = Field(default=True, description="Use LangChain agents")
    use_memory: bool = Field(default=True, description="Use LangChain memory")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")

class BatchProcessorConfig(BaseConfig):
    """Batch processor configuration settings."""
    
    # Processing Settings
    max_workers: int = Field(default=8, description="Maximum worker threads")
    batch_size: int = Field(default=10, description="Batch size")
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    timeout: float = Field(default=300.0, description="Processing timeout")
    
    # Progress Tracking
    enable_progress_tracking: bool = Field(default=True, description="Enable progress tracking")
    progress_reporting_interval: float = Field(default=5.0, description="Progress reporting interval")
    
    # Resource Management
    memory_limit_mb: int = Field(default=1024, description="Memory limit in MB")
    cpu_limit_percent: int = Field(default=80, description="CPU limit percentage")
    
    # Retry Settings
    retry_attempts: int = Field(default=3, description="Retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay")

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

class SecurityConfig(BaseConfig):
    """Security configuration settings."""
    
    # Rate Limiting
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # Input Validation
    enable_input_validation: bool = Field(default=True, description="Enable input validation")
    max_url_length: int = Field(default=500, description="Maximum URL length")
    max_title_length: int = Field(default=200, description="Maximum title length")
    max_description_length: int = Field(default=1000, description="Maximum description length")
    
    # Security Headers
    enable_security_headers: bool = Field(default=True, description="Enable security headers")
    content_security_policy: str = Field(
        default="default-src 'self'",
        description="Content Security Policy"
    )
    
    # Authentication
    enable_authentication: bool = Field(default=True, description="Enable authentication")
    token_expiration_hours: int = Field(default=24, description="Token expiration in hours")
    
    # Malware Detection
    enable_malware_detection: bool = Field(default=True, description="Enable malware detection")
    malware_scan_timeout: int = Field(default=30, description="Malware scan timeout")

# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

class Settings(
    BaseConfig,
    DatabaseConfig,
    RedisConfig,
    CacheConfig,
    MonitoringConfig,
    VideoProcessorConfig,
    ViralProcessorConfig,
    LangChainConfig,
    BatchProcessorConfig,
    SecurityConfig
):
    """Main application settings combining all configuration classes."""
    
    # Environment-specific overrides
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._apply_environment_overrides()
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        if self.environment == Environment.PRODUCTION:
            self.debug = False
            self.reload = False
            self.log_level = LogLevel.INFO
            self.cors_origins = ["https://your-domain.com"]
            self.enable_rate_limiting = True
            self.enable_security_headers = True
        elif self.environment == Environment.DEVELOPMENT:
            self.debug = True
            self.reload = True
            self.log_level = LogLevel.DEBUG
            self.cors_origins = ["*"]
            self.database_echo = True
        elif self.environment == Environment.TESTING:
            self.debug = True
            self.database_url = "sqlite:///./test.db"
            self.redis_db = 1
            self.cache_enabled = False
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING
    
    def get_database_url(self) -> str:
        """Get database URL with proper formatting."""
        if self.is_testing:
            return "sqlite:///./test.db"
        return self.database_url
    
    def get_redis_url(self) -> str:
        """Get Redis URL with proper formatting."""
        return self.redis_url
    
    def get_cache_key(self, *parts: str) -> str:
        """Generate cache key from parts."""
        return self.cache_key_separator.join([self.cache_key_prefix] + list(parts))
    
    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": self.log_format,
                },
                "structured": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": self.log_level.value.upper(),
                "handlers": ["default"],
            },
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# =============================================================================
# CONFIGURATION INSTANCE
# =============================================================================

# Global settings instance
settings = Settings()

# Environment-specific settings
def get_settings() -> Settings:
    """Get application settings."""
    return settings

def get_environment() -> Environment:
    """Get current environment."""
    return settings.environment

def is_production() -> bool:
    """Check if running in production."""
    return settings.is_production

def is_development() -> bool:
    """Check if running in development."""
    return settings.is_development

def is_testing() -> bool:
    """Check if running in testing."""
    return settings.is_testing

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_configuration() -> bool:
    """Validate configuration settings."""
    try:
        # Validate required settings
        if not settings.secret_key or settings.secret_key == "your-secret-key-change-in-production":
            if settings.is_production:
                raise ValueError("Secret key must be set in production")
        
        if not settings.jwt_secret_key or settings.jwt_secret_key == "your-jwt-secret-key-change-in-production":
            if settings.is_production:
                raise ValueError("JWT secret key must be set in production")
        
        # Validate database URL
        if not settings.is_testing:
            if not settings.database_url.startswith(('postgresql://', 'postgresql+asyncpg://')):
                raise ValueError("Invalid database URL format")
        
        # Validate Redis connection
        if settings.cache_enabled:
            if not settings.redis_host:
                raise ValueError("Redis host must be set when caching is enabled")
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'Environment',
    'LogLevel',
    'BaseConfig',
    'DatabaseConfig',
    'RedisConfig',
    'CacheConfig',
    'MonitoringConfig',
    'VideoProcessorConfig',
    'ViralProcessorConfig',
    'LangChainConfig',
    'BatchProcessorConfig',
    'SecurityConfig',
    'Settings',
    'settings',
    'get_settings',
    'get_environment',
    'is_production',
    'is_development',
    'is_testing',
    'validate_configuration'
]






























