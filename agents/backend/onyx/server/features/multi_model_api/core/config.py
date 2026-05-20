"""
Configuration management for Multi-Model API
"""

import os
import logging
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class MultiModelConfig(BaseSettings):
    """Configuration for Multi-Model API with validation"""
    
    # Sentry configuration
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    sentry_environment: str = Field(default="production", description="Sentry environment")
    sentry_traces_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Sentry traces sample rate")
    
    # Cache configuration
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    cache_l1_max_size: int = Field(default=1000, ge=1, le=100000, description="L1 cache max size")
    cache_l1_ttl: int = Field(default=300, ge=1, description="L1 cache TTL in seconds")
    cache_l2_ttl: int = Field(default=3600, ge=1, description="L2 cache TTL in seconds")
    cache_enable_compression: bool = Field(default=True, description="Enable cache compression")
    
    # Rate limiting configuration
    rate_limit_default: int = Field(default=100, ge=1, le=10000, description="Default rate limit")
    rate_limit_window: int = Field(default=60, ge=1, le=3600, description="Rate limit window in seconds")
    rate_limit_burst: int = Field(default=10, ge=1, le=1000, description="Rate limit burst")
    
    # Model configuration
    model_timeout: float = Field(default=30.0, ge=1.0, le=300.0, description="Default model timeout in seconds")
    model_max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    model_retry_delay: float = Field(default=1.0, ge=0.1, le=60.0, description="Retry delay in seconds")
    
    # Feature flags
    enable_prometheus: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_websocket: bool = Field(default=True, description="Enable WebSocket support")
    enable_sentry: bool = Field(default=True, description="Enable Sentry error tracking")
    enable_metrics: bool = Field(default=True, description="Enable internal metrics")
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator('redis_url')
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        """Validate Redis URL format"""
        if not v.startswith(('redis://', 'rediss://')):
            raise ValueError("redis_url must start with redis:// or rediss://")
        return v
    
    def configure_logging(self) -> None:
        """Configure logging based on config"""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured at {self.log_level} level")
    
    class Config:
        env_prefix = "MULTI_MODEL_"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields


_config_instance: Optional[MultiModelConfig] = None


def get_config() -> MultiModelConfig:
    """
    Get or create configuration instance
    
    Returns:
        MultiModelConfig instance
        
    Note:
        Configuration is loaded from environment variables with MULTI_MODEL_ prefix
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = MultiModelConfig()
        # Configure logging on first load
        _config_instance.configure_logging()
    return _config_instance

