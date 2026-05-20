"""
Application configuration
Centralized configuration management
"""

import os
from typing import Optional
from pydantic import Field, field_validator

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older Pydantic versions
    from pydantic import BaseSettings


class AppConfig(BaseSettings):
    """Application configuration"""
    
    # Application
    app_name: str = Field(default="Addiction Recovery AI", description="Application name")
    app_version: str = Field(default="3.3.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="production", description="Environment")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8018, description="Server port")
    
    # CORS
    cors_origins: list[str] = Field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"],
        description="CORS allowed origins"
    )
    cors_allow_credentials: bool = Field(
        default=os.getenv("CORS_ALLOW_CREDENTIALS", "True").lower() == "true",
        description="CORS allow credentials"
    )
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, description="Requests per minute")
    rate_limit_per_hour: int = Field(default=1000, description="Requests per hour")
    
    # Cache
    cache_ttl_default: int = Field(default=300, description="Default cache TTL in seconds")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    
    # Database
    database_url: Optional[str] = Field(default=None, description="Database URL")
    database_pool_size: int = Field(default=10, description="Database pool size")
    
    # Security
    secret_key: Optional[str] = Field(default=None, description="Secret key for JWT")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=60, description="Access token expiration")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get application configuration (singleton)"""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config

