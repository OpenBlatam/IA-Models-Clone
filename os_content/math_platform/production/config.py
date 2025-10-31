from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
from enum import Enum
import secrets
from pydantic import BaseSettings, validator
        import logging.config
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Production Configuration
Environment-based configuration management with validation and security.
"""



class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(Enum):
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
    name: str = "math_platform"
    user: str = "math_user"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    ssl_mode: str = "prefer"
    
    @property
    def connection_string(self) -> str:
        """Get database connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    
    @property
    def connection_string(self) -> str:
        """Get Redis connection string."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    bcrypt_rounds: int = 12
    cors_origins: List[str] = field(default_factory=list)
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    
    def __post_init__(self) -> Any:
        if not self.secret_key:
            self.secret_key = secrets.token_urlsafe(32)


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    health_check_interval: int = 30
    metrics_interval: int = 60
    log_retention_days: int = 30
    alert_email: str = ""
    sentry_dsn: str = ""
    jaeger_endpoint: str = ""
    opentelemetry_enabled: bool = False


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    max_workers: int = 8
    cache_size: int = 10000
    cache_ttl: int = 3600
    batch_size: int = 100
    timeout_seconds: int = 30
    max_concurrent_requests: int = 100
    connection_pool_size: int = 20
    keepalive_timeout: int = 65


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    access_log: bool = True
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    title: str = "Math Platform API"
    version: str = "2.0.0"
    description: str = "Advanced mathematical operations platform"


class ProductionSettings(BaseSettings):
    """Production settings with environment variable support."""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "math_platform"
    db_user: str = "math_user"
    db_password: str = ""
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    
    # Security
    secret_key: str = ""
    cors_origins: str = "*"
    
    # Monitoring
    prometheus_enabled: bool = True
    sentry_dsn: str = ""
    
    # Performance
    max_workers: int = 8
    cache_size: int = 10000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("environment", pre=True)
    def validate_environment(cls, v) -> bool:
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator("log_level", pre=True)
    def validate_log_level(cls, v) -> bool:
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    @validator("cors_origins", pre=True)
    def validate_cors_origins(cls, v) -> bool:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


class ProductionConfig:
    """Production configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        
    """__init__ function."""
self.config_path = config_path
        self.settings = ProductionSettings()
        self._load_config()
        self._validate_config()
        self._setup_logging()
    
    def _load_config(self) -> Any:
        """Load configuration from file if provided."""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, value)
    
    def _validate_config(self) -> bool:
        """Validate configuration settings."""
        if self.settings.environment == Environment.PRODUCTION:
            if not self.settings.secret_key:
                raise ValueError("Secret key is required in production")
            if self.settings.debug:
                raise ValueError("Debug mode should be disabled in production")
            if self.settings.cors_origins == "*":
                raise ValueError("CORS origins should be specific in production")
    
    def _setup_logging(self) -> Any:
        """Setup logging configuration."""
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "json": {
                    "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.settings.log_level.value,
                    "formatter": "default",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": self.settings.log_level.value,
                    "formatter": "json" if self.settings.environment == Environment.PRODUCTION else "default",
                    "filename": f"logs/math_platform_{self.settings.environment.value}.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                }
            },
            "loggers": {
                "math_platform": {
                    "level": self.settings.log_level.value,
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["console"]
            }
        }
        
        logging.config.dictConfig(log_config)
    
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        return DatabaseConfig(
            host=self.settings.db_host,
            port=self.settings.db_port,
            name=self.settings.db_name,
            user=self.settings.db_user,
            password=self.settings.db_password
        )
    
    @property
    def redis(self) -> RedisConfig:
        """Get Redis configuration."""
        return RedisConfig(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            password=self.settings.redis_password,
            db=self.settings.redis_db
        )
    
    @property
    def security(self) -> SecurityConfig:
        """Get security configuration."""
        return SecurityConfig(
            secret_key=self.settings.secret_key,
            cors_origins=self.settings.cors_origins
        )
    
    @property
    def monitoring(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return MonitoringConfig(
            prometheus_enabled=self.settings.prometheus_enabled,
            sentry_dsn=self.settings.sentry_dsn
        )
    
    @property
    def performance(self) -> PerformanceConfig:
        """Get performance configuration."""
        return PerformanceConfig(
            max_workers=self.settings.max_workers,
            cache_size=self.settings.cache_size
        )
    
    @property
    async def api(self) -> APIConfig:
        """Get API configuration."""
        return APIConfig(
            host=self.settings.api_host,
            port=self.settings.api_port,
            workers=self.settings.api_workers
        )
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return {
            "environment": self.settings.environment.value,
            "debug": self.settings.debug,
            "log_level": self.settings.log_level.value,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "name": self.database.name,
                "user": self.database.user,
                "pool_size": self.database.pool_size
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "workers": self.api.workers
            },
            "performance": {
                "max_workers": self.performance.max_workers,
                "cache_size": self.performance.cache_size
            }
        }
    
    def validate_production_ready(self) -> List[str]:
        """Validate if configuration is production-ready."""
        issues = []
        
        if self.settings.environment == Environment.PRODUCTION:
            if not self.settings.secret_key:
                issues.append("Secret key is required in production")
            if self.settings.debug:
                issues.append("Debug mode should be disabled in production")
            if self.settings.cors_origins == "*":
                issues.append("CORS origins should be specific in production")
            if not self.settings.sentry_dsn:
                issues.append("Sentry DSN is recommended for production error tracking")
            if self.settings.db_password == "":
                issues.append("Database password is required in production")
        
        return issues


# Global configuration instance
config = ProductionConfig() 