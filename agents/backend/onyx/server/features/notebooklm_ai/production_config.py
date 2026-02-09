from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Production Configuration
========================

Production configuration settings for the Advanced Library Integration system.
Includes environment-specific settings, security configurations, and deployment options.
"""


class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(Enum):
    """Log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "notebooklm_ai"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create from environment variables"""
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            name=os.getenv("DB_NAME", "notebooklm_ai"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "30")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600"))
        )

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Create from environment variables"""
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD", ""),
            db=int(os.getenv("REDIS_DB", "0")),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "50")),
            socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
            socket_connect_timeout=int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
        )

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    bcrypt_rounds: int = 12
    cors_origins: List[str] = field(default_factory=list)
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600
    
    @classmethod
    def from_env(cls) -> 'SecurityConfig':
        """Create from environment variables"""
        return cls(
            secret_key=os.getenv("SECRET_KEY", ""),
            algorithm=os.getenv("ALGORITHM", "HS256"),
            access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
            refresh_token_expire_days=int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7")),
            bcrypt_rounds=int(os.getenv("BCRYPT_ROUNDS", "12")),
            cors_origins=os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [],
            rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "3600"))
        )

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    grafana_enabled: bool = True
    grafana_port: int = 3000
    log_level: LogLevel = LogLevel.INFO
    log_file: str = "logs/production.log"
    log_max_size: int = 100 * 1024 * 1024  # 100MB
    log_backup_count: int = 5
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> 'MonitoringConfig':
        """Create from environment variables"""
        return cls(
            prometheus_enabled=os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true",
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
            grafana_enabled=os.getenv("GRAFANA_ENABLED", "true").lower() == "true",
            grafana_port=int(os.getenv("GRAFANA_PORT", "3000")),
            log_level=LogLevel(os.getenv("LOG_LEVEL", "info")),
            log_file=os.getenv("LOG_FILE", "logs/production.log"),
            log_max_size=int(os.getenv("LOG_MAX_SIZE", str(100 * 1024 * 1024))),
            log_backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            metrics_enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            tracing_enabled=os.getenv("TRACING_ENABLED", "true").lower() == "true"
        )

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    max_workers: int = 4
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    batch_size: int = 10
    cache_ttl: int = 3600
    gpu_enabled: bool = True
    gpu_memory_fraction: float = 0.8
    model_quantization: bool = True
    async_processing: bool = True
    
    @classmethod
    def from_env(cls) -> 'PerformanceConfig':
        """Create from environment variables"""
        return cls(
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "100")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            batch_size=int(os.getenv("BATCH_SIZE", "10")),
            cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
            gpu_enabled=os.getenv("GPU_ENABLED", "true").lower() == "true",
            gpu_memory_fraction=float(os.getenv("GPU_MEMORY_FRACTION", "0.8")),
            model_quantization=os.getenv("MODEL_QUANTIZATION", "true").lower() == "true",
            async_processing=os.getenv("ASYNC_PROCESSING", "true").lower() == "true"
        )

@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8001
    workers: int = 1
    reload: bool = False
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    docs_enabled: bool = True
    cors_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Create from environment variables"""
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8001")),
            workers=int(os.getenv("API_WORKERS", "1")),
            reload=os.getenv("API_RELOAD", "false").lower() == "true",
            ssl_keyfile=os.getenv("SSL_KEYFILE"),
            ssl_certfile=os.getenv("SSL_CERTFILE"),
            docs_enabled=os.getenv("DOCS_ENABLED", "true").lower() == "true",
            cors_enabled=os.getenv("CORS_ENABLED", "true").lower() == "true"
        )

@dataclass
class StorageConfig:
    """Storage configuration"""
    type: str = "local"  # local, s3, gcs, azure
    local_path: str = "storage"
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    gcs_bucket: str = ""
    azure_container: str = ""
    azure_connection_string: str = ""
    
    @classmethod
    def from_env(cls) -> 'StorageConfig':
        """Create from environment variables"""
        return cls(
            type=os.getenv("STORAGE_TYPE", "local"),
            local_path=os.getenv("STORAGE_LOCAL_PATH", "storage"),
            s3_bucket=os.getenv("S3_BUCKET", ""),
            s3_region=os.getenv("S3_REGION", "us-east-1"),
            s3_access_key=os.getenv("S3_ACCESS_KEY", ""),
            s3_secret_key=os.getenv("S3_SECRET_KEY", ""),
            gcs_bucket=os.getenv("GCS_BUCKET", ""),
            azure_container=os.getenv("AZURE_CONTAINER", ""),
            azure_connection_string=os.getenv("AZURE_CONNECTION_STRING", "")
        )

@dataclass
class ProductionConfig:
    """Main production configuration"""
    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    database: DatabaseConfig = field(default_factory=DatabaseConfig.from_env)
    redis: RedisConfig = field(default_factory=RedisConfig.from_env)
    security: SecurityConfig = field(default_factory=SecurityConfig.from_env)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig.from_env)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig.from_env)
    api: APIConfig = field(default_factory=APIConfig.from_env)
    storage: StorageConfig = field(default_factory=StorageConfig.from_env)
    
    def __post_init__(self) -> Any:
        """Post initialization setup"""
        # Set environment-specific defaults
        if self.environment == Environment.DEVELOPMENT:
            self.debug = True
            self.monitoring.log_level = LogLevel.DEBUG
            self.api.docs_enabled = True
            self.api.reload = True
        elif self.environment == Environment.PRODUCTION:
            self.debug = False
            self.monitoring.log_level = LogLevel.INFO
            self.api.docs_enabled = False
            self.api.reload = False
            self.api.workers = max(1, os.cpu_count() or 1)
    
    @classmethod
    def from_env(cls) -> 'ProductionConfig':
        """Create configuration from environment variables"""
        env_str = os.getenv("ENVIRONMENT", "production").lower()
        environment = Environment(env_str)
        
        return cls(
            environment=environment,
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "name": self.database.name,
                "user": self.database.user,
                "pool_size": self.database.pool_size,
                "max_overflow": self.database.max_overflow
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
                "max_connections": self.redis.max_connections
            },
            "security": {
                "algorithm": self.security.algorithm,
                "access_token_expire_minutes": self.security.access_token_expire_minutes,
                "bcrypt_rounds": self.security.bcrypt_rounds,
                "cors_origins": self.security.cors_origins,
                "rate_limit_requests": self.security.rate_limit_requests
            },
            "monitoring": {
                "log_level": self.monitoring.log_level.value,
                "log_file": self.monitoring.log_file,
                "prometheus_enabled": self.monitoring.prometheus_enabled,
                "metrics_enabled": self.monitoring.metrics_enabled
            },
            "performance": {
                "max_workers": self.performance.max_workers,
                "max_concurrent_requests": self.performance.max_concurrent_requests,
                "batch_size": self.performance.batch_size,
                "gpu_enabled": self.performance.gpu_enabled,
                "async_processing": self.performance.async_processing
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "workers": self.api.workers,
                "docs_enabled": self.api.docs_enabled,
                "cors_enabled": self.api.cors_enabled
            },
            "storage": {
                "type": self.storage.type,
                "local_path": self.storage.local_path
            }
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate required fields
        if not self.security.secret_key:
            errors.append("SECRET_KEY is required")
        
        if self.database.password == "":
            errors.append("DB_PASSWORD is required")
        
        if self.storage.type == "s3" and not self.storage.s3_bucket:
            errors.append("S3_BUCKET is required when using S3 storage")
        
        if self.storage.type == "gcs" and not self.storage.gcs_bucket:
            errors.append("GCS_BUCKET is required when using GCS storage")
        
        if self.storage.type == "azure" and not self.storage.azure_container:
            errors.append("AZURE_CONTAINER is required when using Azure storage")
        
        # Validate port ranges
        if not (1 <= self.api.port <= 65535):
            errors.append("API_PORT must be between 1 and 65535")
        
        if not (1 <= self.database.port <= 65535):
            errors.append("DB_PORT must be between 1 and 65535")
        
        if not (1 <= self.redis.port <= 65535):
            errors.append("REDIS_PORT must be between 1 and 65535")
        
        return errors
    
    def create_directories(self) -> Any:
        """Create necessary directories"""
        directories = [
            "logs",
            "storage",
            "temp",
            "cache",
            "models"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_database_url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.database.user}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.name}"
    
    def get_redis_url(self) -> str:
        """Get Redis URL"""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.db}"
        else:
            return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"

# Global configuration instance
config = ProductionConfig.from_env()

def get_config() -> ProductionConfig:
    """Get the global configuration instance"""
    return config

def validate_config() -> bool:
    """Validate the configuration and return True if valid"""
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    return True

def print_config():
    """Print the current configuration"""
    print("=== Production Configuration ===")
    print(json.dumps(config.to_dict(), indent=2))
    print("================================")

if __name__ == "__main__":
    # Test configuration
    print_config()
    
    if validate_config():
        print("✅ Configuration is valid")
        config.create_directories()
        print("✅ Directories created")
    else:
        print("❌ Configuration has errors")
        exit(1) 