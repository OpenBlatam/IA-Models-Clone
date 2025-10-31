"""
Configuration management using Pydantic Settings.
"""

from functools import lru_cache
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Application
    app_name: str = "Ultra Modular AI History Comparison"
    app_version: str = "8.0.0"
    debug: bool = False
    environment: str = "development"
    
    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: List[str] = ["*"]
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./app.db"
    database_echo: bool = False
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_ttl: int = 3600
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"
    
    # Plugin System
    plugin_directory: str = "plugins"
    plugin_timeout: int = 30
    max_concurrent_plugins: int = 10
    
    # Extension System
    extension_timeout: int = 10
    max_extensions_per_point: int = 100
    
    # Middleware
    middleware_timeout: int = 5
    max_middleware_per_pipeline: int = 50
    
    # Event System
    event_bus_workers: int = 10
    event_queue_size: int = 1000
    
    # Workflow System
    workflow_timeout: int = 3600
    max_workflow_instances: int = 100
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed_envs = ["development", "staging", "production"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of {allowed_envs}")
        return v
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()




