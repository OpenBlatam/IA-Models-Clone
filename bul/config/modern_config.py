"""
Modern Configuration System for BUL
===================================

Using Pydantic Settings for better validation and type safety.
"""

import os
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import logging
from pydantic import BaseSettings, Field, validator, root_validator
from pydantic_settings import BaseSettings as PydanticBaseSettings

logger = logging.getLogger(__name__)

class Environment(str, Enum):
    """Environment types"""
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

class CacheBackend(str, Enum):
    """Cache backend types"""
    MEMORY = "memory"
    REDIS = "redis"

class APIConfig(BaseSettings):
    """API configuration with validation"""
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key (optional)")
    openrouter_base_url: str = Field("https://openrouter.ai/api/v1", description="OpenRouter base URL")
    openai_base_url: str = Field("https://api.openai.com/v1", description="OpenAI base URL")
    default_model: str = Field("openai/gpt-4", description="Default model to use")
    fallback_model: str = Field("openai/gpt-3.5-turbo", description="Fallback model")
    max_tokens: int = Field(4000, ge=100, le=8000, description="Maximum tokens per request")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation")
    timeout: int = Field(30, ge=5, le=300, description="API timeout in seconds")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    
    @validator('openrouter_api_key')
    def validate_openrouter_key(cls, v):
        if not v or len(v) < 10:
            raise ValueError('OpenRouter API key must be provided and valid')
        return v
    
    @validator('default_model', 'fallback_model')
    def validate_model_names(cls, v):
        if not v or '/' not in v:
            raise ValueError('Model name must be in format "provider/model"')
        return v

class DatabaseConfig(BaseSettings):
    """Database configuration"""
    url: str = Field("sqlite:///bul.db", description="Database URL")
    echo: bool = Field(False, description="Echo SQL queries")
    pool_size: int = Field(5, ge=1, le=50, description="Connection pool size")
    max_overflow: int = Field(10, ge=0, le=100, description="Maximum pool overflow")
    pool_timeout: int = Field(30, ge=5, le=300, description="Pool timeout in seconds")
    pool_recycle: int = Field(3600, ge=300, le=86400, description="Pool recycle time in seconds")
    
    @validator('url')
    def validate_database_url(cls, v):
        if not v:
            raise ValueError('Database URL must be provided')
        return v

class ServerConfig(BaseSettings):
    """Server configuration"""
    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(8000, ge=1, le=65535, description="Server port")
    workers: int = Field(1, ge=1, le=32, description="Number of workers")
    reload: bool = Field(False, description="Enable auto-reload")
    log_level: LogLevel = Field(LogLevel.INFO, description="Log level")
    cors_origins: List[str] = Field(["*"], description="CORS allowed origins")
    max_request_size: int = Field(16 * 1024 * 1024, description="Max request size in bytes")  # 16MB
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

class CacheConfig(BaseSettings):
    """Cache configuration"""
    enabled: bool = Field(True, description="Enable caching")
    backend: CacheBackend = Field(CacheBackend.MEMORY, description="Cache backend")
    redis_url: str = Field("redis://localhost:6379/0", description="Redis URL")
    default_ttl: int = Field(3600, ge=60, le=86400, description="Default TTL in seconds")
    max_size: int = Field(1000, ge=10, le=10000, description="Maximum cache size")
    
    @validator('redis_url')
    def validate_redis_url(cls, v, values):
        if values.get('backend') == CacheBackend.REDIS and not v:
            raise ValueError('Redis URL must be provided when using Redis backend')
        return v

class LoggingConfig(BaseSettings):
    """Logging configuration"""
    level: LogLevel = Field(LogLevel.INFO, description="Log level")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    file_path: Optional[str] = Field(None, description="Log file path")
    max_file_size: int = Field(10 * 1024 * 1024, description="Max log file size in bytes")  # 10MB
    backup_count: int = Field(5, ge=1, le=20, description="Number of backup files")
    json_logs: bool = Field(False, description="Use JSON format for logs")

class SecurityConfig(BaseSettings):
    """Security configuration"""
    secret_key: str = Field(..., description="Secret key for JWT tokens")
    algorithm: str = Field("HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(30, ge=5, le=1440, description="Access token expiry")
    refresh_token_expire_days: int = Field(7, ge=1, le=30, description="Refresh token expiry")
    password_min_length: int = Field(8, ge=6, le=128, description="Minimum password length")
    rate_limit_requests: int = Field(100, ge=1, le=10000, description="Rate limit per minute")
    rate_limit_window: int = Field(60, ge=1, le=3600, description="Rate limit window in seconds")
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if not v or len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v

class BULConfig(PydanticBaseSettings):
    """Main BUL configuration using Pydantic Settings"""
    
    # Environment
    environment: Environment = Field(Environment.DEVELOPMENT, description="Environment")
    debug: bool = Field(False, description="Debug mode")
    
    # Sub-configurations
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Business logic configuration
    max_documents_per_batch: int = Field(10, ge=1, le=100, description="Max documents per batch")
    max_document_length: int = Field(50000, ge=1000, le=100000, description="Max document length")
    default_language: str = Field("es", description="Default language")
    supported_languages: List[str] = Field(["es", "en", "pt", "fr"], description="Supported languages")
    
    # Agent configuration
    agent_auto_selection: bool = Field(True, description="Enable automatic agent selection")
    agent_fallback_enabled: bool = Field(True, description="Enable agent fallback")
    max_agent_retries: int = Field(3, ge=1, le=10, description="Max agent retries")
    
    # Document generation configuration
    default_format: str = Field("markdown", description="Default document format")
    supported_formats: List[str] = Field(["markdown", "html", "pdf", "docx"], description="Supported formats")
    quality_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Quality threshold")
    max_processing_time: int = Field(300, ge=30, le=1800, description="Max processing time in seconds")
    
    # Performance configuration
    enable_compression: bool = Field(True, description="Enable response compression")
    enable_caching: bool = Field(True, description="Enable response caching")
    cache_ttl: int = Field(1800, ge=60, le=86400, description="Cache TTL in seconds")
    
    @validator('supported_languages', 'supported_formats', pre=True)
    def parse_list_fields(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(',')]
        return v
    
    @validator('default_language')
    def validate_default_language(cls, v, values):
        supported = values.get('supported_languages', [])
        if supported and v not in supported:
            raise ValueError(f'Default language {v} must be in supported languages: {supported}')
        return v
    
    @validator('default_format')
    def validate_default_format(cls, v, values):
        supported = values.get('supported_formats', [])
        if supported and v not in supported:
            raise ValueError(f'Default format {v} must be in supported formats: {supported}')
        return v
    
    @root_validator
    def validate_environment_settings(cls, values):
        """Validate environment-specific settings"""
        env = values.get('environment')
        debug = values.get('debug')
        
        if env == Environment.PRODUCTION:
            if debug:
                raise ValueError('Debug mode cannot be enabled in production')
            
            # Production-specific validations
            cors_origins = values.get('server', {}).get('cors_origins', [])
            if '*' in cors_origins:
                logger.warning('CORS is set to allow all origins in production')
        
        return values
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_nested_delimiter = "__"
        
        # Example: API__OPENROUTER_API_KEY=your_key
        # This will set api.openrouter_api_key

# Global configuration instance
_config: Optional[BULConfig] = None

def get_config() -> BULConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = BULConfig()
        logger.info(f"Configuration loaded for environment: {_config.environment}")
    return _config

def reload_config() -> BULConfig:
    """Reload configuration from environment"""
    global _config
    _config = BULConfig()
    logger.info(f"Configuration reloaded for environment: {_config.environment}")
    return _config

def validate_config() -> bool:
    """Validate the current configuration"""
    try:
        config = get_config()
        # Pydantic will raise ValidationError if invalid
        logger.info("Configuration validation passed")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

# Configuration utilities
def is_production() -> bool:
    """Check if running in production"""
    return get_config().environment == Environment.PRODUCTION

def is_development() -> bool:
    """Check if running in development"""
    return get_config().environment == Environment.DEVELOPMENT

def is_testing() -> bool:
    """Check if running in testing"""
    return get_config().environment == Environment.TESTING




