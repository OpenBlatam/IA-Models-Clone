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
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import structlog
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Configuration management for OS Content UGC Video Generator
Centralized configuration with environment variable support
"""


logger = structlog.get_logger("os_content.config")

@dataclass
class ServerConfig:
    """Server configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    log_level: str = "info"
    debug: bool = False

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    redis_url: Optional[str] = None
    memory_size: int = 1000
    disk_size: int = 10000
    ttl: int = 3600
    compression: bool = True

@dataclass
class ProcessorConfig:
    """Async processor configuration"""
    max_concurrent: int = 20
    max_workers: int = 4
    throttle_rate: int = 100
    task_timeout: int = 300

@dataclass
class LoadBalancerConfig:
    """Load balancer configuration"""
    backend_servers: List[str] = field(default_factory=list)
    algorithm: str = "round_robin"
    health_check_interval: int = 30

@dataclass
class CDNConfig:
    """CDN configuration"""
    cdn_url: str = ""
    cache_size: int = 1024 * 1024 * 1024  # 1GB
    cache_ttl: int = 3600

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    metrics_save_interval: int = 300
    prometheus_url: str = "http://prometheus:9090"
    grafana_url: str = "http://grafana:3000"

@dataclass
class SecurityConfig:
    """Security configuration"""
    rate_limit: int = 60
    rate_limit_window: int = 60
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

@dataclass
class StorageConfig:
    """Storage configuration"""
    upload_dir: str = "./uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"])

@dataclass
class Config:
    """Main configuration class"""
    server: ServerConfig = field(default_factory=ServerConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    load_balancer: LoadBalancerConfig = field(default_factory=LoadBalancerConfig)
    cdn: CDNConfig = field(default_factory=CDNConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    def __post_init__(self) -> Any:
        """Load configuration from environment variables"""
        self._load_from_env()
        self._validate_config()
        self._create_directories()
    
    def _load_from_env(self) -> Any:
        """Load configuration from environment variables"""
        # Server config
        self.server.host = os.getenv("HOST", self.server.host)
        self.server.port = int(os.getenv("PORT", str(self.server.port)))
        self.server.workers = int(os.getenv("WORKERS", str(self.server.workers)))
        self.server.log_level = os.getenv("LOG_LEVEL", self.server.log_level)
        self.server.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Cache config
        self.cache.redis_url = os.getenv("REDIS_URL", self.cache.redis_url)
        self.cache.memory_size = int(os.getenv("CACHE_MEMORY_SIZE", str(self.cache.memory_size)))
        self.cache.disk_size = int(os.getenv("CACHE_DISK_SIZE", str(self.cache.disk_size)))
        self.cache.ttl = int(os.getenv("CACHE_TTL", str(self.cache.ttl)))
        self.cache.compression = os.getenv("ENABLE_COMPRESSION", "true").lower() == "true"
        
        # Processor config
        self.processor.max_concurrent = int(os.getenv("MAX_CONCURRENT_TASKS", str(self.processor.max_concurrent)))
        self.processor.max_workers = int(os.getenv("MAX_WORKERS", str(self.processor.max_workers)))
        self.processor.throttle_rate = int(os.getenv("THROTTLE_RATE", str(self.processor.throttle_rate)))
        self.processor.task_timeout = int(os.getenv("TASK_TIMEOUT", str(self.processor.task_timeout)))
        
        # Load balancer config
        backend_servers = os.getenv("BACKEND_SERVERS", "")
        if backend_servers:
            self.load_balancer.backend_servers = [s.strip() for s in backend_servers.split(",") if s.strip()]
        self.load_balancer.algorithm = os.getenv("LOAD_BALANCER_ALGORITHM", self.load_balancer.algorithm)
        self.load_balancer.health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", str(self.load_balancer.health_check_interval)))
        
        # CDN config
        self.cdn.cdn_url = os.getenv("CDN_URL", self.cdn.cdn_url)
        self.cdn.cache_size = int(os.getenv("CDN_CACHE_SIZE", str(self.cdn.cache_size)))
        self.cdn.cache_ttl = int(os.getenv("CDN_CACHE_TTL", str(self.cdn.cache_ttl)))
        
        # Monitoring config
        self.monitoring.enabled = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
        self.monitoring.metrics_save_interval = int(os.getenv("METRICS_SAVE_INTERVAL", str(self.monitoring.metrics_save_interval)))
        self.monitoring.prometheus_url = os.getenv("PROMETHEUS_URL", self.monitoring.prometheus_url)
        self.monitoring.grafana_url = os.getenv("GRAFANA_URL", self.monitoring.grafana_url)
        
        # Security config
        self.security.rate_limit = int(os.getenv("RATE_LIMIT", str(self.security.rate_limit)))
        self.security.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", str(self.security.rate_limit_window)))
        self.security.ssl_cert_path = os.getenv("SSL_CERT_PATH", self.security.ssl_cert_path)
        self.security.ssl_key_path = os.getenv("SSL_KEY_PATH", self.security.ssl_key_path)
        
        # Storage config
        self.storage.upload_dir = os.getenv("UPLOAD_DIR", self.storage.upload_dir)
        self.storage.max_file_size = int(os.getenv("MAX_FILE_SIZE", str(self.storage.max_file_size)))
        allowed_extensions = os.getenv("ALLOWED_EXTENSIONS", "")
        if allowed_extensions:
            self.storage.allowed_extensions = [ext.strip() for ext in allowed_extensions.split(",") if ext.strip()]
    
    def _validate_config(self) -> bool:
        """Validate configuration values"""
        if self.server.port < 1 or self.server.port > 65535:
            raise ValueError(f"Invalid port number: {self.server.port}")
        
        if self.processor.max_concurrent < 1:
            raise ValueError(f"Invalid max_concurrent: {self.processor.max_concurrent}")
        
        if self.cache.memory_size < 1:
            raise ValueError(f"Invalid cache memory size: {self.cache.memory_size}")
        
        if self.storage.max_file_size < 1:
            raise ValueError(f"Invalid max file size: {self.storage.max_file_size}")
    
    def _create_directories(self) -> Any:
        """Create necessary directories"""
        directories = [
            self.storage.upload_dir,
            "./logs",
            "./cache",
            "./cdn_cache",
            "./ssl"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers,
                "log_level": self.server.log_level,
                "debug": self.server.debug
            },
            "cache": {
                "redis_url": self.cache.redis_url,
                "memory_size": self.cache.memory_size,
                "disk_size": self.cache.disk_size,
                "ttl": self.cache.ttl,
                "compression": self.cache.compression
            },
            "processor": {
                "max_concurrent": self.processor.max_concurrent,
                "max_workers": self.processor.max_workers,
                "throttle_rate": self.processor.throttle_rate,
                "task_timeout": self.processor.task_timeout
            },
            "load_balancer": {
                "backend_servers": self.load_balancer.backend_servers,
                "algorithm": self.load_balancer.algorithm,
                "health_check_interval": self.load_balancer.health_check_interval
            },
            "cdn": {
                "cdn_url": self.cdn.cdn_url,
                "cache_size": self.cdn.cache_size,
                "cache_ttl": self.cdn.cache_ttl
            },
            "monitoring": {
                "enabled": self.monitoring.enabled,
                "metrics_save_interval": self.monitoring.metrics_save_interval,
                "prometheus_url": self.monitoring.prometheus_url,
                "grafana_url": self.monitoring.grafana_url
            },
            "security": {
                "rate_limit": self.security.rate_limit,
                "rate_limit_window": self.security.rate_limit_window,
                "ssl_cert_path": self.security.ssl_cert_path,
                "ssl_key_path": self.security.ssl_key_path
            },
            "storage": {
                "upload_dir": self.storage.upload_dir,
                "max_file_size": self.storage.max_file_size,
                "allowed_extensions": self.storage.allowed_extensions
            }
        }

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance"""
    return config

def reload_config() -> Config:
    """Reload configuration from environment"""
    global config
    config = Config()
    logger.info("Configuration reloaded")
    return config 