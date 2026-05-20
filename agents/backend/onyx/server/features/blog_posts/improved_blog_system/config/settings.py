"""
Configuration settings for the improved blog system
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation and type hints."""
    
    # API Configuration
    api_title: str = Field(default="Blog System API", description="API title")
    api_version: str = Field(default="1.0.0", description="API version")
    api_description: str = Field(
        default="A modern, scalable blog system built with FastAPI",
        description="API description"
    )
    debug: bool = Field(default=False, description="Debug mode")
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost/blog_db",
        description="Database connection URL"
    )
    database_pool_size: int = Field(default=20, ge=1, le=100, description="Database pool size")
    database_max_overflow: int = Field(default=30, ge=0, le=100, description="Database max overflow")
    database_pool_pre_ping: bool = Field(default=True, description="Database pool pre-ping")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    cache_ttl: int = Field(default=3600, ge=60, description="Cache TTL in seconds")
    
    # Security Configuration
    secret_key: str = Field(..., description="Secret key for JWT tokens")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, ge=1, description="Access token expiration")
    refresh_token_expire_days: int = Field(default=7, ge=1, description="Refresh token expiration")
    
    # CORS Configuration
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    cors_allow_credentials: bool = Field(default=True, description="CORS allow credentials")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=100, ge=1, description="Rate limit per minute")
    
    # File Upload
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max file size in bytes")  # 10MB
    allowed_file_types: List[str] = Field(
        default=["image/jpeg", "image/png", "image/gif", "image/webp"],
        description="Allowed file types"
    )
    
    # AI/ML Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Default ML model name"
    )
    max_sequence_length: int = Field(default=512, ge=128, le=2048, description="Max sequence length")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    log_level: str = Field(default="INFO", description="Logging level")
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    
    # Pagination
    default_page_size: int = Field(default=20, ge=1, le=100, description="Default page size")
    max_page_size: int = Field(default=100, ge=1, le=1000, description="Maximum page size")
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

