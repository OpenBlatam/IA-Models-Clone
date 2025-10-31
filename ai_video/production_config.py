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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Production Configuration for AI Video System

This module provides production configuration management including environment
variables, deployment settings, monitoring, and security configurations.
"""


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "ai_video_production"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    
    @classmethod
    def from_env(cls) -> Any:
        """Create database config from environment variables."""
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "ai_video_production"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20"))
        )

@dataclass
class RedisConfig:
    """Redis configuration settings."""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    database: int = 0
    max_connections: int = 20
    
    @classmethod
    def from_env(cls) -> Any:
        """Create Redis config from environment variables."""
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD", ""),
            database=int(os.getenv("REDIS_DB", "0")),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
        )

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    api_key_required: bool = True
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    rate_limit_per_minute: int = 100
    cors_origins: List[str] = field(default_factory=list)
    allowed_ips: List[str] = field(default_factory=list)
    
    @classmethod
    def from_env(cls) -> Any:
        """Create security config from environment variables."""
        return cls(
            api_key_required=os.getenv("API_KEY_REQUIRED", "true").lower() == "true",
            jwt_secret=os.getenv("JWT_SECRET", ""),
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            jwt_expiration_hours=int(os.getenv("JWT_EXPIRATION_HOURS", "24")),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "100")),
            cors_origins=os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [],
            allowed_ips=os.getenv("ALLOWED_IPS", "").split(",") if os.getenv("ALLOWED_IPS") else []
        )

@dataclass
class MonitoringConfig:
    """Monitoring configuration settings."""
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    health_check_interval: int = 30
    metrics_collection_interval: int = 60
    log_level: str = "INFO"
    log_file: str = "production.log"
    log_rotation_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    
    @classmethod
    def from_env(cls) -> Any:
        """Create monitoring config from environment variables."""
        return cls(
            prometheus_enabled=os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true",
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            metrics_collection_interval=int(os.getenv("METRICS_COLLECTION_INTERVAL", "60")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "production.log"),
            log_rotation_size=int(os.getenv("LOG_ROTATION_SIZE", str(10 * 1024 * 1024))),
            log_backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
        )

@dataclass
class OptimizationConfig:
    """Optimization configuration settings."""
    enable_numba: bool = True
    enable_dask: bool = True
    enable_redis: bool = True
    enable_prometheus: bool = True
    enable_ray: bool = False
    numba_cache_size: int = 1000
    dask_workers: int = 4
    dask_memory_limit: str = "2GB"
    
    @classmethod
    def from_env(cls) -> Any:
        """Create optimization config from environment variables."""
        return cls(
            enable_numba=os.getenv("ENABLE_NUMBA", "true").lower() == "true",
            enable_dask=os.getenv("ENABLE_DASK", "true").lower() == "true",
            enable_redis=os.getenv("ENABLE_REDIS", "true").lower() == "true",
            enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true",
            enable_ray=os.getenv("ENABLE_RAY", "false").lower() == "true",
            numba_cache_size=int(os.getenv("NUMBA_CACHE_SIZE", "1000")),
            dask_workers=int(os.getenv("DASK_WORKERS", "4")),
            dask_memory_limit=os.getenv("DASK_MEMORY_LIMIT", "2GB")
        )

@dataclass
class StorageConfig:
    """Storage configuration settings."""
    cache_dir: str = "cache"
    results_dir: str = "results"
    temp_dir: str = "temp"
    upload_dir: str = "uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])
    
    @classmethod
    def from_env(cls) -> Any:
        """Create storage config from environment variables."""
        return cls(
            cache_dir=os.getenv("CACHE_DIR", "cache"),
            results_dir=os.getenv("RESULTS_DIR", "results"),
            temp_dir=os.getenv("TEMP_DIR", "temp"),
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
            max_file_size=int(os.getenv("MAX_FILE_SIZE", str(100 * 1024 * 1024))),
            allowed_extensions=os.getenv("ALLOWED_EXTENSIONS", ".mp4,.avi,.mov,.mkv").split(",")
        )

@dataclass
class ProductionConfig:
    """Complete production configuration."""
    
    # Environment
    environment: str = "production"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Workflow settings
    max_concurrent_workflows: int = 10
    workflow_timeout: int = 300
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig.from_env)
    redis: RedisConfig = field(default_factory=RedisConfig.from_env)
    security: SecurityConfig = field(default_factory=SecurityConfig.from_env)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig.from_env)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig.from_env)
    storage: StorageConfig = field(default_factory=StorageConfig.from_env)
    
    def __post_init__(self) -> Any:
        """Initialize configuration and create directories."""
        # Set environment
        self.environment = os.getenv("ENVIRONMENT", "production")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Set server settings
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.workers = int(os.getenv("WORKERS", "4"))
        
        # Set workflow settings
        self.max_concurrent_workflows = int(os.getenv("MAX_CONCURRENT_WORKFLOWS", "10"))
        self.workflow_timeout = int(os.getenv("WORKFLOW_TIMEOUT", "300"))
        self.retry_attempts = int(os.getenv("RETRY_ATTEMPTS", "3"))
        self.retry_delay = float(os.getenv("RETRY_DELAY", "1.0"))
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self) -> Any:
        """Create necessary directories."""
        directories = [
            self.storage.cache_dir,
            self.storage.results_dir,
            self.storage.temp_dir,
            self.storage.upload_dir,
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "max_concurrent_workflows": self.max_concurrent_workflows,
            "workflow_timeout": self.workflow_timeout,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username,
                "pool_size": self.database.pool_size,
                "max_overflow": self.database.max_overflow
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "database": self.redis.database,
                "max_connections": self.redis.max_connections
            },
            "security": {
                "api_key_required": self.security.api_key_required,
                "jwt_algorithm": self.security.jwt_algorithm,
                "jwt_expiration_hours": self.security.jwt_expiration_hours,
                "rate_limit_per_minute": self.security.rate_limit_per_minute,
                "cors_origins": self.security.cors_origins,
                "allowed_ips": self.security.allowed_ips
            },
            "monitoring": {
                "prometheus_enabled": self.monitoring.prometheus_enabled,
                "prometheus_port": self.monitoring.prometheus_port,
                "health_check_interval": self.monitoring.health_check_interval,
                "metrics_collection_interval": self.monitoring.metrics_collection_interval,
                "log_level": self.monitoring.log_level,
                "log_file": self.monitoring.log_file
            },
            "optimization": {
                "enable_numba": self.optimization.enable_numba,
                "enable_dask": self.optimization.enable_dask,
                "enable_redis": self.optimization.enable_redis,
                "enable_prometheus": self.optimization.enable_prometheus,
                "enable_ray": self.optimization.enable_ray,
                "numba_cache_size": self.optimization.numba_cache_size,
                "dask_workers": self.optimization.dask_workers,
                "dask_memory_limit": self.optimization.dask_memory_limit
            },
            "storage": {
                "cache_dir": self.storage.cache_dir,
                "results_dir": self.storage.results_dir,
                "temp_dir": self.storage.temp_dir,
                "upload_dir": self.storage.upload_dir,
                "max_file_size": self.storage.max_file_size,
                "allowed_extensions": self.storage.allowed_extensions
            }
        }
    
    def save_to_file(self, filepath: str):
        """Save configuration to file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ProductionConfig':
        """Load configuration from file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_dict = json.load(f)
        
        # Create config instance
        config = cls()
        
        # Update with file values
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate required fields
        if not self.database.host:
            errors.append("Database host is required")
        
        if not self.database.database:
            errors.append("Database name is required")
        
        if self.security.api_key_required and not self.security.jwt_secret:
            errors.append("JWT secret is required when API key is required")
        
        # Validate port ranges
        if not (1 <= self.port <= 65535):
            errors.append("Port must be between 1 and 65535")
        
        if not (1 <= self.database.port <= 65535):
            errors.append("Database port must be between 1 and 65535")
        
        if not (1 <= self.redis.port <= 65535):
            errors.append("Redis port must be between 1 and 65535")
        
        # Validate positive integers
        if self.max_concurrent_workflows <= 0:
            errors.append("Max concurrent workflows must be positive")
        
        if self.workflow_timeout <= 0:
            errors.append("Workflow timeout must be positive")
        
        if self.retry_attempts < 0:
            errors.append("Retry attempts must be non-negative")
        
        return errors

def create_production_config() -> ProductionConfig:
    """Create production configuration with environment variable support."""
    return ProductionConfig()

def validate_production_config(config: ProductionConfig) -> bool:
    """Validate production configuration."""
    errors = config.validate()
    
    if errors:
        logging.error("Configuration validation failed:")
        for error in errors:
            logging.error(f"  - {error}")
        return False
    
    logging.info("Configuration validation passed")
    return True

if __name__ == "__main__":
    # Example usage
    config = create_production_config()
    
    if validate_production_config(config):
        print("Configuration is valid")
        config.save_to_file("production_config.json")
        print("Configuration saved to production_config.json")
    else:
        print("Configuration is invalid")
        sys.exit(1) 