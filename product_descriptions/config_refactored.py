from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from enum import Enum
from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Refactored Configuration Module
==============================

Clean, centralized configuration with environment variables support.
"""



class Environment(str, Enum):
    """Application environments."""
    LOCAL = "local"
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class AppConfig(BaseSettings):
    """Application configuration."""
    
    # App basics
    name: str = Field(default="Refactored Product API")
    version: str = Field(default="2.0.0")
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    
    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)
    
    # API
    api_prefix: str = Field(default="/api/v2")
    docs_url: Optional[str] = Field(default="/docs")
    
    # Database
    database_url: str = Field(default="postgresql+asyncpg://user:pass@localhost/products")
    db_pool_size: int = Field(default=10)
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_max_connections: int = Field(default=20)
    cache_ttl: int = Field(default=3600)
    
    # Security
    secret_key: str = Field(default="change-in-production")
    rate_limit_enabled: bool = Field(default=True)
    requests_per_minute: int = Field(default=100)
    
    # Monitoring
    enable_metrics: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    sentry_dsn: Optional[str] = Field(default=None)
    
    # AI
    openai_api_key: Optional[str] = Field(default=None)
    enable_ai: bool = Field(default=False)
    
    model_config = {"env_file": ".env", "case_sensitive": False}
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT


@lru_cache()
def get_config() -> AppConfig:
    """Get cached configuration."""
    return AppConfig()


# Global config instance
config = get_config() 