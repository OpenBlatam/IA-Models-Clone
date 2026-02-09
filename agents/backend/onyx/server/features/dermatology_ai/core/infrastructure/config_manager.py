"""
Configuration Manager
Centralized configuration management with validation and type safety
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = "sqlite"
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheConfig:
    """Cache configuration"""
    type: str = "memory"
    host: Optional[str] = None
    port: Optional[int] = None
    ttl: int = 3600
    max_size: int = 1000
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    enabled: bool = True
    requests_per_second: float = 10.0
    burst_size: int = 20
    per_user: bool = True
    per_ip: bool = True


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    enabled: bool = True
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class ImageProcessorConfig:
    """Image processor configuration"""
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    min_image_size: int = 1024  # 1KB
    allowed_formats: list = field(default_factory=lambda: ["jpeg", "jpg", "png", "webp"])
    timeout: float = 30.0


@dataclass
class AppConfig:
    """Application configuration"""
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    image_processor: ImageProcessorConfig = field(default_factory=ImageProcessorConfig)
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables"""
        config = cls()
        
        # Environment
        config.environment = os.getenv("ENVIRONMENT", "development")
        config.debug = os.getenv("DEBUG", "false").lower() == "true"
        config.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Database
        config.database.type = os.getenv("DATABASE_TYPE", "sqlite")
        config.database.host = os.getenv("DATABASE_HOST")
        config.database.port = int(os.getenv("DATABASE_PORT", "0")) or None
        config.database.database = os.getenv("DATABASE_NAME")
        config.database.user = os.getenv("DATABASE_USER")
        config.database.password = os.getenv("DATABASE_PASSWORD")
        config.database.pool_size = int(os.getenv("DATABASE_POOL_SIZE", "10"))
        
        # Cache
        config.cache.type = os.getenv("CACHE_TYPE", "memory")
        config.cache.host = os.getenv("CACHE_HOST")
        config.cache.port = int(os.getenv("CACHE_PORT", "0")) or None
        config.cache.ttl = int(os.getenv("CACHE_TTL", "3600"))
        
        # Rate limiting
        config.rate_limit.enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        config.rate_limit.requests_per_second = float(os.getenv("RATE_LIMIT_RPS", "10.0"))
        config.rate_limit.burst_size = int(os.getenv("RATE_LIMIT_BURST", "20"))
        
        # Circuit breaker
        config.circuit_breaker.enabled = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
        config.circuit_breaker.failure_threshold = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
        config.circuit_breaker.timeout = float(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60.0"))
        
        # Image processor
        config.image_processor.max_image_size = int(os.getenv("IMAGE_MAX_SIZE", str(10 * 1024 * 1024)))
        config.image_processor.timeout = float(os.getenv("IMAGE_PROCESSOR_TIMEOUT", "30.0"))
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level,
            "database": {
                "type": self.database.type,
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "pool_size": self.database.pool_size,
            },
            "cache": {
                "type": self.cache.type,
                "host": self.cache.host,
                "port": self.cache.port,
                "ttl": self.cache.ttl,
            },
            "rate_limit": {
                "enabled": self.rate_limit.enabled,
                "requests_per_second": self.rate_limit.requests_per_second,
                "burst_size": self.rate_limit.burst_size,
            },
            "circuit_breaker": {
                "enabled": self.circuit_breaker.enabled,
                "failure_threshold": self.circuit_breaker.failure_threshold,
                "timeout": self.circuit_breaker.timeout,
            },
        }


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
        logger.info(f"Configuration loaded: environment={_config.environment}, debug={_config.debug}")
    return _config


def set_config(config: AppConfig):
    """Set global configuration instance"""
    global _config
    _config = config
    logger.info("Configuration updated")















