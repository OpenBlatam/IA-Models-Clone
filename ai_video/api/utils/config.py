from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
TIMEOUT_SECONDS = 60

from functools import lru_cache
from typing import List
from pydantic import BaseSettings, Field
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuration Management - Pydantic Settings
"""



class CacheSettings(BaseSettings):
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    default_ttl: int = Field(3600, env="CACHE_DEFAULT_TTL")


class CORSSettings(BaseSettings):
    origins: List[str] = Field(["*"], env="CORS_ORIGINS")
    methods: List[str] = Field(["*"], env="CORS_METHODS")
    headers: List[str] = Field(["*"], env="CORS_HEADERS")


class MonitoringSettings(BaseSettings):
    enabled: bool = Field(True, env="MONITORING_ENABLED")
    metrics_path: str = Field("/metrics", env="METRICS_PATH")


class Settings(BaseSettings):
    # App settings
    app_name: str = Field("AI Video API", env="APP_NAME")
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(True, env="DEBUG")
    
    # Security
    jwt_secret: str = Field("your-secret-key", env="JWT_SECRET")
    jwt_expire_minutes: int = Field(60, env="JWT_EXPIRE_MINUTES")
    
    # Sub-settings
    cache: CacheSettings = CacheSettings()
    cors: CORSSettings = CORSSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    @dataclass
class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings() 