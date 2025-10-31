from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import secrets
from typing import List
from pydantic import Field
    from pydantic_settings import BaseSettings
    from pydantic import BaseSettings
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v5.0 - Configuration Module

Modular configuration management for ultra-fast processing.
"""


try:
except ImportError:
    # Fallback for older pydantic versions


class UltraFastConfig(BaseSettings):
    """Ultra-fast configuration with environment variables and validation."""
    
    # API Information
    API_VERSION: str = "5.0.0"
    API_NAME: str = "Instagram Captions API v5.0 - Ultra-Fast Mass Processing"
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8080, env="PORT")
    
    # Security Configuration
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32), env="SECRET_KEY")
    VALID_API_KEYS: List[str] = Field(
        default=["ultra-key-123", "mass-key-456", "speed-key-789"],
        env="VALID_API_KEYS"
    )
    
    # Ultra-fast Performance Limits
    RATE_LIMIT_REQUESTS: int = Field(default=10000, env="RATE_LIMIT_REQUESTS")  # 10k per hour
    RATE_LIMIT_WINDOW: int = Field(default=3600, env="RATE_LIMIT_WINDOW")
    
    # Mass Processing Configuration
    MAX_BATCH_SIZE: int = Field(default=100, env="MAX_BATCH_SIZE")
    BATCH_TIMEOUT: int = Field(default=30, env="BATCH_TIMEOUT")
    
    # Cache Configuration
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour for mass processing
    CACHE_MAX_SIZE: int = Field(default=50000, env="CACHE_MAX_SIZE")  # 50k items
    
    # AI Engine Configuration
    AI_PARALLEL_WORKERS: int = Field(default=20, env="AI_PARALLEL_WORKERS")
    AI_QUALITY_THRESHOLD: float = Field(default=85.0, env="AI_QUALITY_THRESHOLD")
    AI_MIN_PROCESSING_DELAY: float = Field(default=0.01, env="AI_MIN_PROCESSING_DELAY")
    
    # Bonus Configuration
    STYLE_BONUS: int = Field(default=15, env="STYLE_BONUS")
    AUDIENCE_BONUS: int = Field(default=15, env="AUDIENCE_BONUS")
    PRIORITY_BONUS: int = Field(default=20, env="PRIORITY_BONUS")
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = '{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}'
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    CORS_METHODS: List[str] = ["GET", "POST", "DELETE"]
    
    # Middleware Configuration
    GZIP_MINIMUM_SIZE: int = Field(default=500, env="GZIP_MINIMUM_SIZE")
    
    @dataclass
class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            
    """parse_env_var function."""
if field_name in ["VALID_API_KEYS", "CORS_ORIGINS"]:
                return [x.strip() for x in raw_val.split(",") if x.strip()]
            return cls.json_loads(raw_val)
    
    def get_cache_config(self) -> dict:
        """Get cache-specific configuration."""
        return {
            "ttl": self.CACHE_TTL,
            "max_size": self.CACHE_MAX_SIZE
        }
    
    def get_ai_config(self) -> dict:
        """Get AI engine configuration."""
        return {
            "parallel_workers": self.AI_PARALLEL_WORKERS,
            "quality_threshold": self.AI_QUALITY_THRESHOLD,
            "min_processing_delay": self.AI_MIN_PROCESSING_DELAY,
            "style_bonus": self.STYLE_BONUS,
            "audience_bonus": self.AUDIENCE_BONUS,
            "priority_bonus": self.PRIORITY_BONUS
        }
    
    def get_performance_config(self) -> dict:
        """Get performance-related configuration."""
        return {
            "max_batch_size": self.MAX_BATCH_SIZE,
            "batch_timeout": self.BATCH_TIMEOUT,
            "rate_limit_requests": self.RATE_LIMIT_REQUESTS,
            "rate_limit_window": self.RATE_LIMIT_WINDOW
        }
    
    def get_server_config(self) -> dict:
        """Get server configuration for uvicorn."""
        return {
            "host": self.HOST,
            "port": self.PORT,
            "log_level": self.LOG_LEVEL.lower(),
            "access_log": False,
            "server_header": False,
            "date_header": False,
            "loop": "asyncio"
        }


# Global configuration instance
config = UltraFastConfig() 