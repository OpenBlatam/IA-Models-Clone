from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
from typing import List, Optional, Any
from pydantic import BaseSettings, Field, validator
from functools import lru_cache
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Production configuration settings for LinkedIn Posts application.
"""



class Settings(BaseSettings):
    """Production settings configuration."""
    
    # Application settings
    app_name: str = "LinkedIn Posts API"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    
    # Database settings
    database_url: str = Field(env="DATABASE_URL")
    db_pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    
    # Redis settings
    redis_url: str = Field(env="REDIS_URL")
    redis_max_connections: int = Field(default=100, env="REDIS_MAX_CONNECTIONS")
    
    # Cache settings
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    cache_max_size: int = Field(default=10000, env="CACHE_MAX_SIZE")
    
    # Security settings
    secret_key: str = Field(env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    
    # AI/ML settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    max_tokens: int = Field(default=1000, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # LinkedIn API settings
    linkedin_client_id: Optional[str] = Field(default=None, env="LINKEDIN_CLIENT_ID")
    linkedin_client_secret: Optional[str] = Field(default=None, env="LINKEDIN_CLIENT_SECRET")
    linkedin_redirect_uri: Optional[str] = Field(default=None, env="LINKEDIN_REDIRECT_URI")
    
    # Background tasks
    max_background_tasks: int = Field(default=100, env="MAX_BACKGROUND_TASKS")
    task_timeout: int = Field(default=300, env="TASK_TIMEOUT")  # 5 minutes
    
    # File upload settings
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    allowed_file_types: List[str] = Field(
        default=["image/jpeg", "image/png", "image/gif", "video/mp4"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Performance settings
    enable_compression: bool = Field(default=True, env="ENABLE_COMPRESSION")
    compression_level: int = Field(default=6, env="COMPRESSION_LEVEL")
    
    # Health check settings
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Feature flags
    enable_ai_optimization: bool = Field(default=True, env="ENABLE_AI_OPTIMIZATION")
    enable_analytics: bool = Field(default=True, env="ENABLE_ANALYTICS")
    enable_templates: bool = Field(default=True, env="ENABLE_TEMPLATES")
    enable_batch_operations: bool = Field(default=True, env="ENABLE_BATCH_OPERATIONS")
    
    @validator('cors_origins', pre=True)
    def assemble_cors_origins(cls, v) -> Any:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    @validator('allowed_file_types', pre=True)
    def assemble_allowed_file_types(cls, v) -> Any:
        """Parse allowed file types from string or list."""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    @validator('database_url')
    def validate_database_url(cls, v) -> bool:
        """Validate database URL."""
        if not v:
            raise ValueError("DATABASE_URL is required")
        return v
    
    @validator('redis_url')
    def validate_redis_url(cls, v) -> bool:
        """Validate Redis URL."""
        if not v:
            raise ValueError("REDIS_URL is required")
        return v
    
    @validator('secret_key')
    def validate_secret_key(cls, v) -> bool:
        """Validate secret key."""
        if not v:
            raise ValueError("SECRET_KEY is required")
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v
    
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
class DevelopmentSettings(Settings):
    """Development environment settings."""
    debug: bool = True
    workers: int = 1
    log_level: str = "DEBUG"
    enable_metrics: bool = False
    enable_tracing: bool = False


class ProductionSettings(Settings):
    """Production environment settings."""
    debug: bool = False
    workers: int = 4
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_tracing: bool = True
    
    # Production-specific security
    cors_origins: List[str] = []  # Must be explicitly set
    
    @validator('cors_origins')
    def validate_production_cors(cls, v) -> bool:
        """Validate CORS origins in production."""
        if not v or v == ["*"]:
            raise ValueError("CORS_ORIGINS must be explicitly set in production")
        return v


class TestSettings(Settings):
    """Test environment settings."""
    debug: bool = True
    database_url: str = "sqlite+aiosqlite:///:memory:"
    redis_url: str = "redis://localhost:6379/1"
    secret_key: str = "test-secret-key-for-testing-only-32-chars"
    enable_metrics: bool = False
    enable_tracing: bool = False


def get_settings_for_environment(env: str = None) -> Settings:
    """Get settings based on environment."""
    if env is None:
        env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return ProductionSettings()
    elif env == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()


# Export default settings
settings = get_settings_for_environment() 