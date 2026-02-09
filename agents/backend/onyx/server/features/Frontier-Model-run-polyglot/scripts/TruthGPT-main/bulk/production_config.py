#!/usr/bin/env python3
"""
Production Configuration - Production-ready configuration management
Handles environment-specific configurations, secrets, and deployment settings
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum
import secrets
import string

class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(Enum):
    """Log levels."""
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
    database: str = "bulk_optimization"
    username: str = "bulk_user"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    ssl_mode: str = "prefer"
    connection_timeout: int = 30

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

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    enable_grafana: bool = True
    grafana_port: int = 3000
    enable_health_checks: bool = True
    health_check_interval: int = 30
    enable_metrics_collection: bool = True
    metrics_retention_days: int = 30

@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = ""
    jwt_secret: str = ""
    jwt_expiration: int = 3600
    enable_cors: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    enable_ssl: bool = True
    ssl_cert_path: str = ""
    ssl_key_path: str = ""

@dataclass
class PerformanceConfig:
    """Performance configuration."""
    max_workers: int = 4
    max_memory_gb: float = 16.0
    max_cpu_usage: float = 80.0
    enable_gpu_acceleration: bool = True
    gpu_memory_fraction: float = 0.8
    enable_mixed_precision: bool = True
    batch_size: int = 32
    prefetch_factor: int = 2
    enable_async_processing: bool = True
    async_timeout: int = 300

@dataclass
class ProductionConfig:
    """Production configuration."""
    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "bulk_optimization.log"
    log_max_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Application settings
    app_name: str = "Bulk Optimization System"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Optimization settings
    optimization_timeout: int = 3600
    max_concurrent_operations: int = 10
    operation_queue_size: int = 100
    enable_operation_persistence: bool = True
    persistence_directory: str = "/var/lib/bulk_optimization"
    
    # Data processing
    max_file_size_mb: int = 100
    allowed_file_types: list = field(default_factory=lambda: [".json", ".pkl", ".h5", ".zarr"])
    temp_directory: str = "/tmp/bulk_optimization"
    
    # Backup and recovery
    enable_backups: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 7
    backup_directory: str = "/var/backups/bulk_optimization"

class ProductionConfigManager:
    """Production configuration manager."""
    
    def __init__(self, config_file: Optional[str] = None, environment: Optional[Environment] = None):
        self.config_file = config_file
        self.environment = environment or self._detect_environment()
        self.config = self._load_config()
        self._setup_logging()
    
    def _detect_environment(self) -> Environment:
        """Detect environment from environment variables."""
        env = os.getenv("ENVIRONMENT", "development").lower()
        try:
            return Environment(env)
        except ValueError:
            return Environment.DEVELOPMENT
    
    def _load_config(self) -> ProductionConfig:
        """Load configuration from file or environment."""
        if self.config_file and Path(self.config_file).exists():
            return self._load_from_file()
        else:
            return self._load_from_environment()
    
    def _load_from_file(self) -> ProductionConfig:
        """Load configuration from file."""
        config_path = Path(self.config_file)
        
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        elif config_path.suffix in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return self._create_config_from_dict(config_data)
    
    def _load_from_environment(self) -> ProductionConfig:
        """Load configuration from environment variables."""
        config = ProductionConfig()
        
        # Environment settings
        config.environment = self.environment
        config.debug = os.getenv("DEBUG", "false").lower() == "true"
        config.log_level = LogLevel(os.getenv("LOG_LEVEL", "INFO"))
        
        # Database settings
        config.database.host = os.getenv("DB_HOST", config.database.host)
        config.database.port = int(os.getenv("DB_PORT", config.database.port))
        config.database.database = os.getenv("DB_NAME", config.database.database)
        config.database.username = os.getenv("DB_USER", config.database.username)
        config.database.password = os.getenv("DB_PASSWORD", config.database.password)
        
        # Redis settings
        config.redis.host = os.getenv("REDIS_HOST", config.redis.host)
        config.redis.port = int(os.getenv("REDIS_PORT", config.redis.port))
        config.redis.password = os.getenv("REDIS_PASSWORD", config.redis.password)
        
        # Security settings
        config.security.secret_key = os.getenv("SECRET_KEY", self._generate_secret_key())
        config.security.jwt_secret = os.getenv("JWT_SECRET", self._generate_secret_key())
        
        # Performance settings
        config.performance.max_workers = int(os.getenv("MAX_WORKERS", config.performance.max_workers))
        config.performance.max_memory_gb = float(os.getenv("MAX_MEMORY_GB", config.performance.max_memory_gb))
        
        # Application settings
        config.host = os.getenv("HOST", config.host)
        config.port = int(os.getenv("PORT", config.port))
        config.workers = int(os.getenv("WORKERS", config.workers))
        
        return config
    
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> ProductionConfig:
        """Create configuration from dictionary."""
        config = ProductionConfig()
        
        # Update with provided data
        for key, value in config_data.items():
            if hasattr(config, key):
                if isinstance(value, dict) and hasattr(getattr(config, key), '__dataclass_fields__'):
                    # Update nested dataclass
                    nested_config = getattr(config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(config, key, value)
        
        return config
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.value),
            format=self.config.log_format,
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler()
            ]
        )
        
        # Setup log rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.config.log_file,
            maxBytes=self.config.log_max_size,
            backupCount=self.config.log_backup_count
        )
        file_handler.setFormatter(logging.Formatter(self.config.log_format))
        
        logger = logging.getLogger()
        logger.addHandler(file_handler)
    
    def get_config(self) -> ProductionConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def save_config(self, filepath: str):
        """Save configuration to file."""
        config_dict = self._config_to_dict()
        
        with open(filepath, 'w') as f:
            if filepath.endswith('.json'):
                json.dump(config_dict, f, indent=2)
            elif filepath.endswith(('.yml', '.yaml')):
                yaml.dump(config_dict, f, default_flow_style=False)
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        
        for field_name, field_value in self.config.__dict__.items():
            if hasattr(field_value, '__dataclass_fields__'):
                # Convert nested dataclass to dict
                nested_dict = {}
                for nested_field_name, nested_field_value in field_value.__dict__.items():
                    nested_dict[nested_field_name] = nested_field_value
                config_dict[field_name] = nested_dict
            else:
                config_dict[field_name] = field_value
        
        return config_dict
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        try:
            # Validate required fields
            if not self.config.security.secret_key:
                raise ValueError("Secret key is required")
            
            if not self.config.database.host:
                raise ValueError("Database host is required")
            
            # Validate numeric fields
            if self.config.port <= 0 or self.config.port > 65535:
                raise ValueError("Invalid port number")
            
            if self.config.performance.max_workers <= 0:
                raise ValueError("Max workers must be positive")
            
            if self.config.performance.max_memory_gb <= 0:
                raise ValueError("Max memory must be positive")
            
            return True
            
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            return False
    
    def get_database_url(self) -> str:
        """Get database URL."""
        return f"postgresql://{self.config.database.username}:{self.config.database.password}@{self.config.database.host}:{self.config.database.port}/{self.config.database.database}"
    
    def get_redis_url(self) -> str:
        """Get Redis URL."""
        if self.config.redis.password:
            return f"redis://:{self.config.redis.password}@{self.config.redis.host}:{self.config.redis.port}/{self.config.redis.db}"
        else:
            return f"redis://{self.config.redis.host}:{self.config.redis.port}/{self.config.redis.db}"

def create_production_config(config_file: Optional[str] = None, 
                           environment: Optional[Environment] = None) -> ProductionConfigManager:
    """Create production configuration manager."""
    return ProductionConfigManager(config_file, environment)

def load_production_config(config_file: str) -> ProductionConfig:
    """Load production configuration from file."""
    manager = ProductionConfigManager(config_file)
    return manager.get_config()

if __name__ == "__main__":
    # Example usage
    config_manager = create_production_config()
    config = config_manager.get_config()
    
    print(f"Environment: {config.environment.value}")
    print(f"Debug: {config.debug}")
    print(f"Log Level: {config.log_level.value}")
    print(f"Database: {config.database.host}:{config.database.port}")
    print(f"Redis: {config.redis.host}:{config.redis.port}")
    print(f"Workers: {config.workers}")
    print(f"Max Memory: {config.performance.max_memory_gb}GB")

