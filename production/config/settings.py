from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
from typing import List, Optional
from functools import lru_cache
from pydantic import BaseSettings, Field, validator
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Production Settings Configuration
================================

Environment-based configuration with validation for production deployment.
"""




class Settings(BaseSettings):
    """Production settings with environment-based configuration."""
    
    # Application
    app_name: str = Field(default="Devin Copywriting API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="production", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    use_uvloop: bool = Field(default=True, env="USE_UVLOOP")
    use_httptools: bool = Field(default=True, env="USE_HTTPTOOLS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    access_log: bool = Field(default=True, env="ACCESS_LOG")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Documentation
    enable_docs: bool = Field(default=True, env="ENABLE_DOCS")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    cors_credentials: bool = Field(default=True, env="CORS_CREDENTIALS")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_pool_size: int = Field(default=20, env="REDIS_POOL_SIZE")
    redis_max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    
    # AI/ML
    ai_model_name: str = Field(default="gpt2", env="AI_MODEL_NAME")
    ai_model_cache_dir: str = Field(default="/tmp/models", env="AI_MODEL_CACHE_DIR")
    ai_max_length: int = Field(default=512, env="AI_MAX_LENGTH")
    ai_temperature: float = Field(default=0.7, env="AI_TEMPERATURE")
    
    # Caching
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    cache_max_size: int = Field(default=10000, env="CACHE_MAX_SIZE")
    enable_predictive_cache: bool = Field(default=True, env="ENABLE_PREDICTIVE_CACHE")
    
    # Performance
    max_workers: int = Field(default=10, env="MAX_WORKERS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    enable_compression: bool = Field(default=True, env="ENABLE_COMPRESSION")
    enable_serialization_optimization: bool = Field(default=True, env="ENABLE_SERIALIZATION_OPTIMIZATION")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Health checks
    health_check_timeout: int = Field(default=5, env="HEALTH_CHECK_TIMEOUT")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=1000, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")
    
    # Circuit breaker
    circuit_breaker_threshold: int = Field(default=5, env="CIRCUIT_BREAKER_THRESHOLD")
    circuit_breaker_timeout: int = Field(default=60, env="CIRCUIT_BREAKER_TIMEOUT")
    
    # External APIs
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_api_base: str = Field(default="https://api.openai.com/v1", env="OPENAI_API_BASE")
    
    # File storage
    upload_dir: str = Field(default="/tmp/uploads", env="UPLOAD_DIR")
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    
    # Email
    smtp_host: Optional[str] = Field(default=None, env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    
    # Third-party services
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    datadog_api_key: Optional[str] = Field(default=None, env="DATADOG_API_KEY")
    
    @validator("environment")
    def validate_environment(cls, v) -> bool:
        """Validate environment setting."""
        valid_environments = ["development", "staging", "production", "test"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v) -> bool:
        """Validate log level setting."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("ai_temperature")
    def validate_temperature(cls, v) -> bool:
        """Validate AI temperature setting."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("AI temperature must be between 0.0 and 2.0")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment.lower() == "test"
    
    def get_cors_config(self) -> dict:
        """Get CORS configuration."""
        return {
            "allow_origins": self.allowed_origins,
            "allow_credentials": self.cors_credentials,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
    
    def get_security_headers(self) -> dict:
        """Get security headers configuration."""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        
        if self.is_production:
            headers.update({
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Content-Security-Policy": "default-src 'self'",
            })
        
        return headers
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Environment-specific settings
def get_development_settings() -> Settings:
    """Get development settings."""
    os.environ.setdefault("ENVIRONMENT", "development")
    os.environ.setdefault("DEBUG", "true")
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    return get_settings()


def get_test_settings() -> Settings:
    """Get test settings."""
    os.environ.setdefault("ENVIRONMENT", "test")
    os.environ.setdefault("DEBUG", "true")
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    return get_settings()


def get_production_settings() -> Settings:
    """Get production settings."""
    os.environ.setdefault("ENVIRONMENT", "production")
    os.environ.setdefault("DEBUG", "false")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    return get_settings() 