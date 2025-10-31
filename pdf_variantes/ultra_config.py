"""Ultra-efficient configuration with minimal overhead."""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class UltraFastConfig:
    """Ultra-fast configuration with minimal overhead."""
    
    # Core settings
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Performance settings
    max_concurrent_requests: int = 1000
    request_timeout: float = 5.0
    cache_ttl: int = 300
    max_file_size_mb: int = 100
    max_batch_size: int = 50
    
    # Caching settings
    enable_caching: bool = True
    cache_max_size: int = 1000
    min_compression_size: int = 1024
    
    # Rate limiting
    max_requests_per_minute: int = 1000
    window_seconds: int = 60
    
    # CORS settings
    cors_origins: list = None
    
    # Feature flags
    features: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]
        
        if self.features is None:
            self.features = {
                "pdf_upload": True,
                "variant_generation": True,
                "topic_extraction": True,
                "batch_processing": True,
                "caching": True,
                "compression": True
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "max_concurrent_requests": self.max_concurrent_requests,
            "request_timeout": self.request_timeout,
            "cache_ttl": self.cache_ttl,
            "max_file_size_mb": self.max_file_size_mb,
            "max_batch_size": self.max_batch_size,
            "enable_caching": self.enable_caching,
            "cache_max_size": self.cache_max_size,
            "min_compression_size": self.min_compression_size,
            "max_requests_per_minute": self.max_requests_per_minute,
            "window_seconds": self.window_seconds,
            "cors_origins": self.cors_origins,
            "features": self.features
        }


class UltraFastConfigManager:
    """Ultra-fast configuration manager."""
    
    def __init__(self):
        self._config: Optional[UltraFastConfig] = None
    
    def load_config(self) -> UltraFastConfig:
        """Load configuration with minimal overhead."""
        if self._config is None:
            self._config = self._load_from_environment()
        
        return self._config
    
    def _load_from_environment(self) -> UltraFastConfig:
        """Load configuration from environment variables."""
        return UltraFastConfig(
            environment=Environment(os.getenv("ENVIRONMENT", "development")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "1000")),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "5.0")),
            cache_ttl=int(os.getenv("CACHE_TTL", "300")),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "100")),
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "50")),
            enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
            cache_max_size=int(os.getenv("CACHE_MAX_SIZE", "1000")),
            min_compression_size=int(os.getenv("MIN_COMPRESSION_SIZE", "1024")),
            max_requests_per_minute=int(os.getenv("MAX_REQUESTS_PER_MINUTE", "1000")),
            window_seconds=int(os.getenv("WINDOW_SECONDS", "60")),
            cors_origins=os.getenv("CORS_ORIGINS", "*").split(",")
        )
    
    def get_config(self) -> UltraFastConfig:
        """Get current configuration."""
        return self.load_config()
    
    def update_config(self, **kwargs) -> None:
        """Update configuration."""
        if self._config is None:
            self._config = self.load_config()
        
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if feature is enabled."""
        config = self.get_config()
        return config.features.get(feature_name, False)
    
    def enable_feature(self, feature_name: str) -> None:
        """Enable a feature."""
        config = self.get_config()
        config.features[feature_name] = True
    
    def disable_feature(self, feature_name: str) -> None:
        """Disable a feature."""
        config = self.get_config()
        config.features[feature_name] = False


# Global configuration manager
config_manager = UltraFastConfigManager()


@lru_cache(maxsize=1)
def get_ultra_fast_config() -> UltraFastConfig:
    """Get ultra-fast configuration with caching."""
    return config_manager.get_config()


def get_environment_config() -> Dict[str, Any]:
    """Get environment-specific configuration."""
    config = get_ultra_fast_config()
    
    env_configs = {
        "development": {
            "debug": True,
            "max_concurrent_requests": 100,
            "cache_ttl": 60,
            "enable_caching": True
        },
        "staging": {
            "debug": False,
            "max_concurrent_requests": 500,
            "cache_ttl": 300,
            "enable_caching": True
        },
        "production": {
            "debug": False,
            "max_concurrent_requests": 1000,
            "cache_ttl": 600,
            "enable_caching": True
        },
        "testing": {
            "debug": True,
            "max_concurrent_requests": 10,
            "cache_ttl": 0,
            "enable_caching": False
        }
    }
    
    return env_configs.get(config.environment.value, {})


def get_performance_config() -> Dict[str, Any]:
    """Get performance-specific configuration."""
    config = get_ultra_fast_config()
    
    return {
        "max_concurrent_requests": config.max_concurrent_requests,
        "request_timeout": config.request_timeout,
        "cache_ttl": config.cache_ttl,
        "max_file_size_mb": config.max_file_size_mb,
        "max_batch_size": config.max_batch_size,
        "enable_caching": config.enable_caching,
        "cache_max_size": config.cache_max_size,
        "min_compression_size": config.min_compression_size
    }


def get_security_config() -> Dict[str, Any]:
    """Get security-specific configuration."""
    config = get_ultra_fast_config()
    
    return {
        "max_requests_per_minute": config.max_requests_per_minute,
        "window_seconds": config.window_seconds,
        "cors_origins": config.cors_origins
    }


def get_feature_config() -> Dict[str, bool]:
    """Get feature-specific configuration."""
    config = get_ultra_fast_config()
    return config.features.copy()


def validate_config(config: UltraFastConfig) -> bool:
    """Validate configuration."""
    if config.max_concurrent_requests <= 0:
        return False
    
    if config.request_timeout <= 0:
        return False
    
    if config.cache_ttl < 0:
        return False
    
    if config.max_file_size_mb <= 0:
        return False
    
    if config.max_batch_size <= 0:
        return False
    
    return True


def get_optimized_config() -> UltraFastConfig:
    """Get optimized configuration for current environment."""
    config = get_ultra_fast_config()
    env_config = get_environment_config()
    
    # Apply environment-specific optimizations
    for key, value in env_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
