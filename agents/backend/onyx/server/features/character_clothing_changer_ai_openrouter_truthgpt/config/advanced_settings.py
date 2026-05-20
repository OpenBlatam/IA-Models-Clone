"""
Advanced Settings
================

Advanced configuration settings for the service.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CacheStrategy(str, Enum):
    """Cache strategies"""
    LRU = "lru"
    FIFO = "fifo"
    LFU = "lfu"


@dataclass
class AdvancedSettings:
    """Advanced settings configuration"""
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    log_max_bytes: int = 10485760  # 10MB
    log_backup_count: int = 5
    
    # Cache
    cache_enabled: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.LRU
    cache_max_size: int = 1000
    cache_default_ttl: float = 3600.0
    cache_cleanup_interval: float = 300.0  # 5 minutes
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_default: int = 100
    rate_limit_window: float = 60.0
    rate_limit_per_client: bool = True
    
    # Performance
    performance_tracking_enabled: bool = True
    performance_max_metrics: int = 1000
    performance_cleanup_interval: float = 3600.0  # 1 hour
    
    # Webhooks
    webhook_timeout: float = 30.0
    webhook_max_retries: int = 3
    webhook_retry_delay: float = 1.0
    webhook_retry_backoff: float = 2.0
    
    # Batch Processing
    batch_max_concurrent_default: int = 5
    batch_max_items: int = 1000
    batch_cleanup_after_hours: float = 24.0
    
    # Health Checks
    health_check_interval: float = 60.0  # 1 minute
    health_check_timeout: float = 10.0
    
    # Metrics
    metrics_retention_days: int = 7
    metrics_cleanup_interval: float = 3600.0  # 1 hour
    
    # Security
    enable_cors: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])
    enable_api_key_auth: bool = False
    api_key_header: str = "X-API-Key"
    
    # Monitoring
    enable_prometheus: bool = False
    prometheus_port: int = 9090
    enable_sentry: bool = False
    sentry_dsn: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "AdvancedSettings":
        """
        Load settings from environment variables.
        
        Returns:
            AdvancedSettings instance
        """
        return cls(
            log_level=LogLevel(os.getenv("LOG_LEVEL", "INFO")),
            log_file=os.getenv("LOG_FILE"),
            cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            cache_max_size=int(os.getenv("CACHE_MAX_SIZE", "1000")),
            rate_limit_enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            rate_limit_default=int(os.getenv("RATE_LIMIT_DEFAULT", "100")),
            performance_tracking_enabled=os.getenv("PERFORMANCE_TRACKING_ENABLED", "true").lower() == "true",
            enable_cors=os.getenv("ENABLE_CORS", "true").lower() == "true",
            enable_api_key_auth=os.getenv("ENABLE_API_KEY_AUTH", "false").lower() == "true",
            enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "false").lower() == "true",
            enable_sentry=os.getenv("ENABLE_SENTRY", "false").lower() == "true",
            sentry_dsn=os.getenv("SENTRY_DSN")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "log_level": self.log_level.value,
            "cache_enabled": self.cache_enabled,
            "cache_max_size": self.cache_max_size,
            "rate_limit_enabled": self.rate_limit_enabled,
            "rate_limit_default": self.rate_limit_default,
            "performance_tracking_enabled": self.performance_tracking_enabled,
            "enable_cors": self.enable_cors,
            "enable_api_key_auth": self.enable_api_key_auth,
            "enable_prometheus": self.enable_prometheus,
            "enable_sentry": self.enable_sentry
        }


def get_advanced_settings() -> AdvancedSettings:
    """
    Get advanced settings instance.
    
    Returns:
        AdvancedSettings instance
    """
    return AdvancedSettings.from_env()

