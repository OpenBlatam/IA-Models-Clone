from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import os
from typing import List, Optional, Dict, Any
from functools import lru_cache
from pydantic import BaseModel, Field, validator
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuration management for Instagram Captions API.

Centralized configuration with environment variables, validation, and type safety.
"""



class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CacheConfig(BaseModel):
    """Cache configuration settings."""
    
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL"
    )
    default_ttl: int = Field(
        default=300,
        ge=1,
        description="Default cache TTL in seconds"
    )
    max_connections: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum Redis connections"
    )
    enabled: bool = Field(
        default=True,
        description="Whether caching is enabled"
    )


class SecurityConfig(BaseModel):
    """Security configuration settings."""
    
    allowed_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    max_request_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        description="Maximum request size in bytes"
    )
    rate_limit_requests: int = Field(
        default=100,
        ge=1,
        description="Rate limit requests per window"
    )
    rate_limit_window: int = Field(
        default=3600,
        ge=1,
        description="Rate limit window in seconds"
    )
    require_api_key: bool = Field(
        default=False,
        description="Whether API key is required"
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="API key header name"
    )


class PerformanceConfig(BaseModel):
    """Performance configuration settings."""
    
    slow_request_threshold: float = Field(
        default=2.0,
        ge=0.1,
        description="Slow request threshold in seconds"
    )
    max_concurrent_requests: int = Field(
        default=100,
        ge=1,
        description="Maximum concurrent requests"
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Request timeout in seconds"
    )
    enable_compression: bool = Field(
        default=True,
        description="Whether to enable response compression"
    )
    enable_metrics: bool = Field(
        default=True,
        description="Whether to enable performance metrics"
    )


class AIProviderConfig(BaseModel):
    """AI provider configuration."""
    
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    openai_model: str = Field(
        default="gpt-4",
        description="OpenAI model to use"
    )
    openrouter_api_key: Optional[str] = Field(
        default=None,
        description="OpenRouter API key"
    )
    langchain_provider: str = Field(
        default="openai",
        description="Default LangChain provider"
    )
    max_tokens: int = Field(
        default=2000,
        ge=100,
        le=8000,
        description="Maximum tokens per request"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="AI model temperature"
    )


class Settings(BaseModel):
    """Application settings with validation."""
    
    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    debug: bool = Field(
        default=False,
        description="Debug mode"
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    
    # API Configuration
    api_title: str = Field(
        default="Instagram Captions API",
        description="API title"
    )
    api_version: str = Field(
        default="2.0.0",
        description="API version"
    )
    api_description: str = Field(
        default="Advanced Instagram caption generation with quality optimization",
        description="API description"
    )
    
    # Server Configuration
    host: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port"
    )
    reload: bool = Field(
        default=False,
        description="Auto-reload on code changes"
    )
    
    # Sub-configurations
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Cache configuration"
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance configuration"
    )
    ai_providers: AIProviderConfig = Field(
        default_factory=AIProviderConfig,
        description="AI provider configuration"
    )
    
    @validator('environment', pre=True)
    def validate_environment(cls, v) -> bool:
        """Validate environment setting."""
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                return Environment.DEVELOPMENT
        return v
    
    @validator('log_level', pre=True)
    def validate_log_level(cls, v) -> bool:
        """Validate log level setting."""
        if isinstance(v, str):
            try:
                return LogLevel(v.upper())
            except ValueError:
                return LogLevel.INFO
        return v
    
    @dataclass
class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


def load_settings_from_env() -> Settings:
    """Load settings from environment variables."""
    
    # Cache configuration
    cache_config = CacheConfig(
        redis_url=os.getenv("REDIS_URL"),
        default_ttl=int(os.getenv("CACHE_DEFAULT_TTL", "300")),
        max_connections=int(os.getenv("CACHE_MAX_CONNECTIONS", "20")),
        enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true"
    )
    
    # Security configuration
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    security_config = SecurityConfig(
        allowed_origins=allowed_origins,
        max_request_size=int(os.getenv("MAX_REQUEST_SIZE", str(10 * 1024 * 1024))),
        rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
        rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "3600")),
        require_api_key=os.getenv("REQUIRE_API_KEY", "false").lower() == "true",
        api_key_header=os.getenv("API_KEY_HEADER", "X-API-Key")
    )
    
    # Performance configuration
    performance_config = PerformanceConfig(
        slow_request_threshold=float(os.getenv("SLOW_REQUEST_THRESHOLD", "2.0")),
        max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "100")),
        timeout_seconds=float(os.getenv("TIMEOUT_SECONDS", "30.0")),
        enable_compression=os.getenv("ENABLE_COMPRESSION", "true").lower() == "true",
        enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true"
    )
    
    # AI provider configuration
    ai_config = AIProviderConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        langchain_provider=os.getenv("LANGCHAIN_PROVIDER", "openai"),
        max_tokens=int(os.getenv("AI_MAX_TOKENS", "2000")),
        temperature=float(os.getenv("AI_TEMPERATURE", "0.7"))
    )
    
    return Settings(
        environment=os.getenv("ENVIRONMENT", "development"),
        debug=os.getenv("DEBUG", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        api_title=os.getenv("API_TITLE", "Instagram Captions API"),
        api_version=os.getenv("API_VERSION", "2.0.0"),
        api_description=os.getenv("API_DESCRIPTION", "Advanced Instagram caption generation"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        cache=cache_config,
        security=security_config,
        performance=performance_config,
        ai_providers=ai_config
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return load_settings_from_env()


def get_logging_config(settings: Settings) -> Dict[str, Any]:
    """Get logging configuration based on settings."""
    
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
            },
            "json": {
                "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
            }
        },
        "handlers": {
            "console": {
                "level": settings.log_level.value,
                "class": "logging.StreamHandler",
                "formatter": "standard" if settings.environment != Environment.PRODUCTION else "json"
            },
            "file": {
                "level": "INFO",
                "class": "logging.FileHandler",
                "filename": "logs/api.log",
                "formatter": "detailed"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console"] if settings.environment == Environment.DEVELOPMENT else ["console", "file"],
                "level": settings.log_level.value,
                "propagate": False
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            }
        }
    }


def get_cors_config(settings: Settings) -> Dict[str, Any]:
    """Get CORS configuration."""
    
    return {
        "allow_origins": settings.security.allowed_origins,
        "allow_credentials": False,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Content-Type",
            "Authorization",
            "X-Request-ID",
            settings.security.api_key_header
        ]
    }


def validate_environment_variables() -> List[str]:
    """Validate required environment variables."""
    
    errors = []
    
    # Check critical environment variables
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        errors.append("Either OPENAI_API_KEY or OPENROUTER_API_KEY must be set")
    
    # Validate numeric environment variables
    numeric_vars = {
        "PORT": (1, 65535),
        "CACHE_DEFAULT_TTL": (1, 86400),
        "RATE_LIMIT_REQUESTS": (1, 10000),
        "TIMEOUT_SECONDS": (1, 300)
    }
    
    for var_name, (min_val, max_val) in numeric_vars.items():
        value = os.getenv(var_name)
        if value:
            try:
                num_val = int(value) if var_name != "TIMEOUT_SECONDS" else float(value)
                if not min_val <= num_val <= max_val:
                    errors.append(f"{var_name} must be between {min_val} and {max_val}")
            except ValueError:
                errors.append(f"{var_name} must be a valid number")
    
    return errors


# Global settings instance
config = get_settings() 