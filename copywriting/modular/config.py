from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
TIMEOUT_SECONDS = 60

import os
import multiprocessing as mp
from typing import Dict, Any, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Modular Configuration Management.

Clean configuration with environment variables and intelligent defaults.
"""


class ModularConfig(BaseSettings):
    """Modular configuration with clean defaults."""
    
    # === API SETTINGS ===
    api_key: str = Field(
        default="modular-copywriting-2024",
        description="API authentication key"
    )
    api_version: str = Field(
        default="v1",
        description="API version"
    )
    
    # === SERVER SETTINGS ===
    host: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    port: int = Field(
        default=8000,
        description="Server port"
    )
    workers: int = Field(
        default_factory=lambda: min(16, mp.cpu_count() * 2),
        description="Number of workers"
    )
    
    # === PERFORMANCE SETTINGS ===
    max_variants: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum variants per request"
    )
    request_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Request timeout in seconds"
    )
    
    # === CACHE SETTINGS ===
    enable_cache: bool = Field(
        default=True,
        description="Enable caching"
    )
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/5",
        description="Redis connection URL"
    )
    
    # === FEATURE FLAGS ===
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    enable_translation: bool = Field(
        default=True,
        description="Enable translation features"
    )
    enable_batch_processing: bool = Field(
        default=True,
        description="Enable batch processing"
    )
    
    # === DEBUG SETTINGS ===
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    @dataclass
class Config:
        env_prefix = "COPYWRITING_"
        env_file = ".env"
        case_sensitive = False
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance configuration info."""
        return {
            "workers": self.workers,
            "max_variants": self.max_variants,
            "request_timeout": self.request_timeout,
            "cache_enabled": self.enable_cache,
            "cache_ttl": self.cache_ttl
        }
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags."""
        return {
            "metrics": self.enable_metrics,
            "rate_limiting": self.enable_rate_limiting,
            "translation": self.enable_translation,
            "batch_processing": self.enable_batch_processing
        }
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug and self.log_level in ["INFO", "WARNING", "ERROR"]

@lru_cache(maxsize=1)
def get_config() -> ModularConfig:
    """Get cached configuration instance."""
    return ModularConfig()

# Export configuration
__all__ = ["ModularConfig", "get_config"] 