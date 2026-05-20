from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import os
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v13.0 - Modular Configuration

Centralized configuration management for modular architecture.
"""



class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "captions_db"
    username: str = "captions_user"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class CacheConfig:
    """Cache configuration."""
    max_size: int = 10000
    default_ttl: int = 3600
    enable_distributed: bool = False
    redis_url: Optional[str] = None
    compression_enabled: bool = True


@dataclass
class AIConfig:
    """AI provider configuration."""
    default_provider: str = "transformers"
    model_name: str = "distilgpt2"
    max_tokens: int = 150
    temperature: float = 0.8
    top_p: float = 0.9
    enable_gpu: bool = False
    fallback_enabled: bool = True
    
    # Provider-specific settings
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    max_workers: int = 10
    request_timeout: float = 30.0
    batch_size_limit: int = 100
    concurrent_requests_limit: int = 50
    enable_jit: bool = True
    enable_vectorization: bool = True


@dataclass
class SecurityConfig:
    """Security configuration."""
    api_keys: List[str] = field(default_factory=lambda: ["modular-v13-key"])
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600
    jwt_secret: Optional[str] = None
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    enable_audit_logging: bool = True
    enable_health_checks: bool = True
    metrics_retention_days: int = 30
    log_level: LogLevel = LogLevel.INFO
    enable_tracing: bool = False


@dataclass
class ModularSettings:
    """Main modular settings configuration."""
    
    # Core settings
    api_version: str = "13.0.0"
    api_name: str = "Instagram Captions API v13.0 - Modular Architecture"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8130
    workers: int = 1
    
    # Module configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Feature flags
    enable_advanced_features: bool = True
    enable_enterprise_features: bool = False
    enable_experimental_features: bool = False
    
    @classmethod
    def from_env(cls) -> "ModularSettings":
        """Create settings from environment variables."""
        settings = cls()
        
        # Override with environment variables
        settings.environment = Environment(os.getenv("ENVIRONMENT", "development"))
        settings.debug = os.getenv("DEBUG", "false").lower() == "true"
        settings.host = os.getenv("HOST", "0.0.0.0")
        settings.port = int(os.getenv("PORT", "8130"))
        
        # AI settings
        settings.ai.openai_api_key = os.getenv("OPENAI_API_KEY")
        settings.ai.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        settings.ai.enable_gpu = os.getenv("ENABLE_GPU", "false").lower() == "true"
        
        # Cache settings
        settings.cache.redis_url = os.getenv("REDIS_URL")
        settings.cache.max_size = int(os.getenv("CACHE_MAX_SIZE", "10000"))
        
        # Performance settings
        settings.performance.max_workers = int(os.getenv("MAX_WORKERS", "10"))
        settings.performance.concurrent_requests_limit = int(os.getenv("CONCURRENT_LIMIT", "50"))
        
        # Security settings
        api_keys_env = os.getenv("API_KEYS")
        if api_keys_env:
            settings.security.api_keys = api_keys_env.split(",")
        
        return settings
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        return (
            f"postgresql://{self.database.username}:{self.database.password}"
            f"@{self.database.host}:{self.database.port}/{self.database.database}"
        )
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration as dict."""
        return {
            "max_size": self.cache.max_size,
            "default_ttl": self.cache.default_ttl,
            "enable_distributed": self.cache.enable_distributed,
            "redis_url": self.cache.redis_url
        }
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI configuration as dict."""
        return {
            "default_provider": self.ai.default_provider,
            "model_name": self.ai.model_name,
            "max_tokens": self.ai.max_tokens,
            "temperature": self.ai.temperature,
            "enable_gpu": self.ai.enable_gpu,
            "fallback_enabled": self.ai.fallback_enabled
        }
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT


# Global settings instance
settings = ModularSettings.from_env()


def get_settings() -> ModularSettings:
    """Get global settings instance."""
    return settings


def reload_settings() -> ModularSettings:
    """Reload settings from environment."""
    global settings
    settings = ModularSettings.from_env()
    return settings 