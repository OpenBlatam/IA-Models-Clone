from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Production Configuration for OS Content UGC Video Generator
Environment-specific settings for production deployment
"""


@dataclass
class ProductionConfig:
    """Production configuration settings"""
    
    # Environment
    environment: str = "production"
    debug: bool = False
    log_level: str = "info"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    worker_class: str = "uvicorn.workers.UvicornWorker"
    
    # Database Configuration
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/os_content"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 50
    redis_socket_timeout: int = 5
    redis_socket_connect_timeout: int = 5
    
    # Cache Configuration
    cache_memory_size: int = 10000
    cache_disk_size: int = 100000
    cache_ttl: int = 3600
    cache_compression: bool = True
    
    # Processing Configuration
    max_concurrent_tasks: int = 50
    max_workers: int = 8
    task_timeout: int = 600
    throttle_rate: int = 200
    
    # File Storage Configuration
    upload_dir: str = "/var/lib/os_content/uploads"
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    allowed_extensions: list = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".gif", ".bmp",
        ".mp4", ".avi", ".mov", ".wmv", ".flv",
        ".mp3", ".wav", ".aac", ".ogg"
    ])
    
    # CDN Configuration
    cdn_url: str = "https://cdn.example.com"
    cdn_cache_size: int = 10 * 1024 * 1024 * 1024  # 10GB
    cdn_cache_ttl: int = 86400  # 24 hours
    
    # Security Configuration
    secret_key: str = "your-secret-key-here"
    jwt_secret: str = "your-jwt-secret-here"
    rate_limit: int = 100
    rate_limit_window: int = 60
    cors_origins: list = field(default_factory=lambda: ["https://example.com"])
    
    # SSL Configuration
    ssl_cert_path: Optional[str] = "/etc/ssl/certs/os_content.crt"
    ssl_key_path: Optional[str] = "/etc/ssl/private/os_content.key"
    
    # Monitoring Configuration
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    grafana_enabled: bool = True
    grafana_port: int = 3000
    metrics_save_interval: int = 60
    
    # Logging Configuration
    log_file: str = "/var/log/os_content/app.log"
    log_max_size: int = 100 * 1024 * 1024  # 100MB
    log_backup_count: int = 5
    log_format: str = "json"
    
    # Backup Configuration
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    backup_retention_days: int = 30
    backup_path: str = "/var/backups/os_content"
    
    # Health Check Configuration
    health_check_interval: int = 30
    health_check_timeout: int = 10
    health_check_retries: int = 3
    
    # Performance Configuration
    enable_gzip: bool = True
    enable_compression: bool = True
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    request_timeout: int = 300
    
    def __post_init__(self) -> Any:
        """Load configuration from environment variables"""
        self._load_from_env()
        self._validate_config()
        self._create_directories()
    
    def _load_from_env(self) -> Any:
        """Load configuration from environment variables"""
        # Environment
        self.environment = os.getenv("ENVIRONMENT", self.environment)
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        
        # Server
        self.host = os.getenv("HOST", self.host)
        self.port = int(os.getenv("PORT", str(self.port)))
        self.workers = int(os.getenv("WORKERS", str(self.workers)))
        self.worker_class = os.getenv("WORKER_CLASS", self.worker_class)
        
        # Database
        self.database_url = os.getenv("DATABASE_URL", self.database_url)
        self.database_pool_size = int(os.getenv("DATABASE_POOL_SIZE", str(self.database_pool_size)))
        self.database_max_overflow = int(os.getenv("DATABASE_MAX_OVERFLOW", str(self.database_max_overflow)))
        self.database_pool_timeout = int(os.getenv("DATABASE_POOL_TIMEOUT", str(self.database_pool_timeout)))
        self.database_pool_recycle = int(os.getenv("DATABASE_POOL_RECYCLE", str(self.database_pool_recycle)))
        
        # Redis
        self.redis_url = os.getenv("REDIS_URL", self.redis_url)
        self.redis_max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", str(self.redis_max_connections)))
        self.redis_socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", str(self.redis_socket_timeout)))
        self.redis_socket_connect_timeout = int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", str(self.redis_socket_connect_timeout)))
        
        # Cache
        self.cache_memory_size = int(os.getenv("CACHE_MEMORY_SIZE", str(self.cache_memory_size)))
        self.cache_disk_size = int(os.getenv("CACHE_DISK_SIZE", str(self.cache_disk_size)))
        self.cache_ttl = int(os.getenv("CACHE_TTL", str(self.cache_ttl)))
        self.cache_compression = os.getenv("CACHE_COMPRESSION", "true").lower() == "true"
        
        # Processing
        self.max_concurrent_tasks = int(os.getenv("MAX_CONCURRENT_TASKS", str(self.max_concurrent_tasks)))
        self.max_workers = int(os.getenv("MAX_WORKERS", str(self.max_workers)))
        self.task_timeout = int(os.getenv("TASK_TIMEOUT", str(self.task_timeout)))
        self.throttle_rate = int(os.getenv("THROTTLE_RATE", str(self.throttle_rate)))
        
        # File Storage
        self.upload_dir = os.getenv("UPLOAD_DIR", self.upload_dir)
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", str(self.max_file_size)))
        allowed_extensions = os.getenv("ALLOWED_EXTENSIONS", "")
        if allowed_extensions:
            self.allowed_extensions = [ext.strip() for ext in allowed_extensions.split(",") if ext.strip()]
        
        # CDN
        self.cdn_url = os.getenv("CDN_URL", self.cdn_url)
        self.cdn_cache_size = int(os.getenv("CDN_CACHE_SIZE", str(self.cdn_cache_size)))
        self.cdn_cache_ttl = int(os.getenv("CDN_CACHE_TTL", str(self.cdn_cache_ttl)))
        
        # Security
        self.secret_key = os.getenv("SECRET_KEY", self.secret_key)
        self.jwt_secret = os.getenv("JWT_SECRET", self.jwt_secret)
        self.rate_limit = int(os.getenv("RATE_LIMIT", str(self.rate_limit)))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", str(self.rate_limit_window)))
        cors_origins = os.getenv("CORS_ORIGINS", "")
        if cors_origins:
            self.cors_origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
        
        # SSL
        self.ssl_cert_path = os.getenv("SSL_CERT_PATH", self.ssl_cert_path)
        self.ssl_key_path = os.getenv("SSL_KEY_PATH", self.ssl_key_path)
        
        # Monitoring
        self.prometheus_enabled = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
        self.prometheus_port = int(os.getenv("PROMETHEUS_PORT", str(self.prometheus_port)))
        self.grafana_enabled = os.getenv("GRAFANA_ENABLED", "true").lower() == "true"
        self.grafana_port = int(os.getenv("GRAFANA_PORT", str(self.grafana_port)))
        self.metrics_save_interval = int(os.getenv("METRICS_SAVE_INTERVAL", str(self.metrics_save_interval)))
        
        # Logging
        self.log_file = os.getenv("LOG_FILE", self.log_file)
        self.log_max_size = int(os.getenv("LOG_MAX_SIZE", str(self.log_max_size)))
        self.log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", str(self.log_backup_count)))
        self.log_format = os.getenv("LOG_FORMAT", self.log_format)
        
        # Backup
        self.backup_enabled = os.getenv("BACKUP_ENABLED", "true").lower() == "true"
        self.backup_schedule = os.getenv("BACKUP_SCHEDULE", self.backup_schedule)
        self.backup_retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", str(self.backup_retention_days)))
        self.backup_path = os.getenv("BACKUP_PATH", self.backup_path)
        
        # Health Check
        self.health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", str(self.health_check_interval)))
        self.health_check_timeout = int(os.getenv("HEALTH_CHECK_TIMEOUT", str(self.health_check_timeout)))
        self.health_check_retries = int(os.getenv("HEALTH_CHECK_RETRIES", str(self.health_check_retries)))
        
        # Performance
        self.enable_gzip = os.getenv("ENABLE_GZIP", "true").lower() == "true"
        self.enable_compression = os.getenv("ENABLE_COMPRESSION", "true").lower() == "true"
        self.max_request_size = int(os.getenv("MAX_REQUEST_SIZE", str(self.max_request_size)))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", str(self.request_timeout)))
    
    def _validate_config(self) -> bool:
        """Validate configuration values"""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid port number: {self.port}")
        
        if self.workers < 1:
            raise ValueError(f"Invalid number of workers: {self.workers}")
        
        if self.max_concurrent_tasks < 1:
            raise ValueError(f"Invalid max concurrent tasks: {self.max_concurrent_tasks}")
        
        if self.max_file_size < 1:
            raise ValueError(f"Invalid max file size: {self.max_file_size}")
        
        if not self.secret_key or self.secret_key == "your-secret-key-here":
            raise ValueError("SECRET_KEY must be set in production")
        
        if not self.jwt_secret or self.jwt_secret == "your-jwt-secret-here":
            raise ValueError("JWT_SECRET must be set in production")
    
    def _create_directories(self) -> Any:
        """Create necessary directories"""
        directories = [
            self.upload_dir,
            Path(self.log_file).parent,
            self.backup_path,
            "/var/cache/os_content",
            "/var/tmp/os_content"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level,
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "worker_class": self.worker_class,
            "database_url": self.database_url.replace(self.database_url.split("@")[0].split("://")[1], "***"),
            "redis_url": self.redis_url.replace("redis://", "redis://***@") if "@" in self.redis_url else self.redis_url,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "max_workers": self.max_workers,
            "upload_dir": self.upload_dir,
            "max_file_size": self.max_file_size,
            "cdn_url": self.cdn_url,
            "rate_limit": self.rate_limit,
            "prometheus_enabled": self.prometheus_enabled,
            "grafana_enabled": self.grafana_enabled,
            "backup_enabled": self.backup_enabled
        }

# Global production configuration instance
production_config = ProductionConfig()

def get_production_config() -> ProductionConfig:
    """Get the global production configuration instance"""
    return production_config 