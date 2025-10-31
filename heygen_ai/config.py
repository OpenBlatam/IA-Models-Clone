from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
from typing import List, Optional, Any, Dict
from pathlib import Path
from pydantic import BaseSettings, Field, validator, root_validator
import structlog
                import json
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Unified Configuration Management

Single source of truth for all application configuration with environment-specific overrides.
"""


logger = structlog.get_logger()


class Settings(BaseSettings):
    """
    Unified application settings with environment-specific configuration.
    
    All configuration is centralized here with proper validation and type safety.
    """
    
    # Application
    app_name: str = Field(default="HeyGen AI API", env="APP_NAME")
    app_version: str = Field(default="2.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    reload: bool = Field(default=False, env="RELOAD")
    
    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./heygen_ai.db",
        env="DATABASE_URL"
    )
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    
    # Security
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    jwt_secret: str = Field(default="dev-jwt-secret-change-in-production", env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # File Storage
    storage_type: str = Field(default="local", env="STORAGE_TYPE")  # local, s3, gcs
    storage_path: str = Field(default="./storage", env="STORAGE_PATH")
    max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    
    # Video Processing
    default_video_quality: str = Field(default="medium", env="DEFAULT_VIDEO_QUALITY")
    max_video_duration_seconds: int = Field(default=300, env="MAX_VIDEO_DURATION_SECONDS")
    max_concurrent_processing: int = Field(default=5, env="MAX_CONCURRENT_PROCESSING")
    
    # External APIs
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    huggingface_token: Optional[str] = Field(default=None, env="HUGGINGFACE_TOKEN")
    
    # Monitoring
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Performance
    enable_gzip: bool = Field(default=True, env="ENABLE_GZIP")
    gzip_minimum_size: int = Field(default=1000, env="GZIP_MINIMUM_SIZE")
    cache_ttl_seconds: int = Field(default=300, env="CACHE_TTL_SECONDS")  # 5 minutes
    
    @dataclass
class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        valid_environments = ["development", "testing", "staging", "production"]
        if v.lower() not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v.lower()
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("storage_type")
    def validate_storage_type(cls, v: str) -> str:
        """Validate storage type."""
        valid_types = ["local", "s3", "gcs"]
        if v.lower() not in valid_types:
            raise ValueError(f"Storage type must be one of: {valid_types}")
        return v.lower()
    
    @validator("default_video_quality")
    def validate_video_quality(cls, v: str) -> str:
        """Validate video quality."""
        valid_qualities = ["low", "medium", "high", "ultra"]
        if v.lower() not in valid_qualities:
            raise ValueError(f"Video quality must be one of: {valid_qualities}")
        return v.lower()
    
    @validator("cors_origins", pre=True)
    def validate_cors_origins(cls, v: Any) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                # Parse JSON-like array string
                return json.loads(v)
            else:
                # Split comma-separated string
                return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    @root_validator
    def validate_production_settings(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate production-specific requirements."""
        environment = values.get("environment", "development")
        
        if environment == "production":
            # Ensure secure settings in production
            if values.get("secret_key") == "dev-secret-key-change-in-production":
                raise ValueError("Must set secure SECRET_KEY in production")
            
            if values.get("jwt_secret") == "dev-jwt-secret-change-in-production":
                raise ValueError("Must set secure JWT_SECRET in production")
            
            if values.get("debug", False):
                logger.warning("Debug mode is enabled in production")
            
            # Ensure CORS is properly configured in production
            cors_origins = values.get("cors_origins", [])
            if "*" in cors_origins:
                logger.warning("CORS allows all origins in production")
        
        return values
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == "testing"
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "pool_timeout": self.database_pool_timeout,
            "echo": self.is_development,
        }
    
    @property
    def redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration dictionary."""
        return {
            "url": self.redis_url,
            "max_connections": self.redis_max_connections,
            "decode_responses": True,
        }
    
    @property
    def cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration dictionary."""
        return {
            "allow_origins": self.cors_origins,
            "allow_credentials": self.cors_allow_credentials,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
    
    @property
    def jwt_config(self) -> Dict[str, Any]:
        """Get JWT configuration dictionary."""
        return {
            "secret": self.jwt_secret,
            "algorithm": self.jwt_algorithm,
            "expiration_hours": self.jwt_expiration_hours,
        }
    
    def get_storage_path(self, relative_path: str = "") -> Path:
        """Get full storage path."""
        base_path = Path(self.storage_path)
        if relative_path:
            return base_path / relative_path
        return base_path
    
    def setup_storage_directories(self) -> None:
        """Create storage directories if they don't exist."""
        if self.storage_type == "local":
            directories = [
                "videos",
                "avatars", 
                "audio",
                "temp",
                "logs"
            ]
            
            for directory in directories:
                dir_path = self.get_storage_path(directory)
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created storage directory: {dir_path}")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get application settings singleton.
    
    Returns the same Settings instance throughout the application lifecycle.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        logger.info(
            "Settings loaded",
            environment=_settings.environment,
            app_name=_settings.app_name,
            version=_settings.app_version
        )
        
        # Setup storage directories if using local storage
        if _settings.storage_type == "local":
            _settings.setup_storage_directories()
    
    return _settings


def reload_settings() -> Settings:
    """Reload settings (useful for testing)."""
    global _settings
    _settings = None
    return get_settings()


# Environment-specific configuration overrides
class DevelopmentSettings(Settings):
    """Development environment settings."""
    environment: str = "development"
    debug: bool = True
    reload: bool = True
    log_level: str = "DEBUG"
    

class TestingSettings(Settings):
    """Testing environment settings."""
    environment: str = "testing"
    database_url: str = "sqlite+aiosqlite:///:memory:"
    redis_url: str = "redis://localhost:6379/1"  # Different DB for tests
    

class ProductionSettings(Settings):
    """Production environment settings."""
    environment: str = "production"
    debug: bool = False
    reload: bool = False
    log_level: str = "INFO"
    workers: int = 4


def get_settings_for_environment(env: str) -> Settings:
    """Get settings for specific environment."""
    settings_map = {
        "development": DevelopmentSettings,
        "testing": TestingSettings, 
        "production": ProductionSettings,
    }
    
    settings_class = settings_map.get(env.lower(), Settings)
    return settings_class() 