#!/usr/bin/env python3
"""
Enhanced Production Configuration - Advanced configuration management
Enhanced with caching, validation, hot-reloading, and advanced security features
"""

import os
import json
import yaml
import hashlib
import secrets
import string
from typing import Dict, Any, Optional, Union, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum
import threading
import time
from datetime import datetime, timedelta
import asyncio
import aiofiles
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

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

class SecurityLevel(Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CacheConfig:
    """Cache configuration."""
    enable_redis_cache: bool = True
    enable_memory_cache: bool = True
    cache_ttl: int = 3600  # seconds
    max_cache_size: int = 1000
    cache_compression: bool = True
    cache_encryption: bool = True
    cache_partitioning: bool = True
    cache_eviction_policy: str = "lru"

@dataclass
class SecurityConfig:
    """Enhanced security configuration."""
    secret_key: str = ""
    jwt_secret: str = ""
    jwt_expiration: int = 3600
    jwt_algorithm: str = "HS256"
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    enable_ssl: bool = True
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    enable_encryption: bool = True
    encryption_key: str = ""
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 90
    enable_ip_whitelist: bool = False
    whitelisted_ips: List[str] = field(default_factory=list)
    enable_2fa: bool = False
    session_timeout: int = 1800
    max_login_attempts: int = 5
    lockout_duration: int = 900
    password_policy: Dict[str, Any] = field(default_factory=lambda: {
        "min_length": 12,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_numbers": True,
        "require_symbols": True,
        "max_age_days": 90
    })

@dataclass
class PerformanceConfig:
    """Enhanced performance configuration."""
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
    enable_connection_pooling: bool = True
    pool_size: int = 20
    pool_overflow: int = 30
    pool_timeout: int = 30
    enable_query_optimization: bool = True
    enable_indexing: bool = True
    enable_compression: bool = True
    compression_level: int = 6
    enable_caching: bool = True
    cache_size_mb: int = 512
    enable_prefetching: bool = True
    prefetch_size: int = 100

@dataclass
class MonitoringConfig:
    """Enhanced monitoring configuration."""
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    enable_grafana: bool = True
    grafana_port: int = 3000
    enable_health_checks: bool = True
    health_check_interval: int = 30
    enable_metrics_collection: bool = True
    metrics_retention_days: int = 30
    enable_alerting: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    enable_tracing: bool = True
    tracing_endpoint: str = ""
    enable_profiling: bool = True
    profiling_interval: int = 60
    enable_anomaly_detection: bool = True
    anomaly_threshold: float = 2.0
    enable_custom_metrics: bool = True
    custom_metrics_interval: int = 10

@dataclass
class EnhancedProductionConfig:
    """Enhanced production configuration."""
    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "bulk_optimization.log"
    log_max_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    enable_structured_logging: bool = True
    enable_log_aggregation: bool = True
    log_aggregation_endpoint: str = ""
    
    # Enhanced component configurations
    cache: CacheConfig = field(default_factory=CacheConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Application settings
    app_name: str = "Bulk Optimization System"
    app_version: str = "2.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    enable_graceful_shutdown: bool = True
    shutdown_timeout: int = 30
    
    # Optimization settings
    optimization_timeout: int = 3600
    max_concurrent_operations: int = 10
    operation_queue_size: int = 100
    enable_operation_persistence: bool = True
    persistence_directory: str = "/var/lib/bulk_optimization"
    enable_operation_encryption: bool = True
    
    # Data processing
    max_file_size_mb: int = 100
    allowed_file_types: List[str] = field(default_factory=lambda: [".json", ".pkl", ".h5", ".zarr"])
    temp_directory: str = "/tmp/bulk_optimization"
    enable_data_validation: bool = True
    enable_data_encryption: bool = True
    data_retention_days: int = 30
    
    # Backup and recovery
    enable_backups: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 7
    backup_directory: str = "/var/backups/bulk_optimization"
    enable_incremental_backups: bool = True
    enable_backup_encryption: bool = True
    
    # Advanced features
    enable_machine_learning: bool = True
    enable_auto_scaling: bool = True
    enable_load_balancing: bool = True
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    enable_retry_mechanism: bool = True
    max_retry_attempts: int = 3
    retry_delay: int = 5

class ConfigValidator:
    """Advanced configuration validator."""
    
    def __init__(self):
        self.validators = {}
        self.custom_validators = {}
    
    def register_validator(self, field_name: str, validator: Callable):
        """Register custom validator for a field."""
        self.custom_validators[field_name] = validator
    
    def validate_config(self, config: EnhancedProductionConfig) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Basic validation
        if config.port <= 0 or config.port > 65535:
            errors.append("Invalid port number")
        
        if config.workers <= 0:
            errors.append("Workers must be positive")
        
        if config.performance.max_memory_gb <= 0:
            errors.append("Max memory must be positive")
        
        # Security validation
        if not config.security.secret_key:
            errors.append("Secret key is required")
        
        if len(config.security.secret_key) < 32:
            errors.append("Secret key must be at least 32 characters")
        
        # Performance validation
        if config.performance.max_workers <= 0:
            errors.append("Max workers must be positive")
        
        if config.performance.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        # Custom validators
        for field_name, validator in self.custom_validators.items():
            try:
                if not validator(getattr(config, field_name, None)):
                    errors.append(f"Custom validation failed for {field_name}")
            except Exception as e:
                errors.append(f"Validation error for {field_name}: {e}")
        
        return errors

class ConfigEncryption:
    """Configuration encryption utilities."""
    
    def __init__(self, password: str):
        self.password = password.encode()
        self.salt = os.urandom(16)
        self.key = self._derive_key()
        self.cipher = Fernet(self.key)
    
    def _derive_key(self) -> bytes:
        """Derive encryption key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(self.password))
    
    def encrypt_config(self, config_dict: Dict[str, Any]) -> str:
        """Encrypt configuration dictionary."""
        config_json = json.dumps(config_dict)
        encrypted_data = self.cipher.encrypt(config_json.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_config(self, encrypted_config: str) -> Dict[str, Any]:
        """Decrypt configuration."""
        encrypted_data = base64.urlsafe_b64decode(encrypted_config.encode())
        decrypted_data = self.cipher.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())

class ConfigHotReloader:
    """Configuration hot-reloading system."""
    
    def __init__(self, config_file: str, callback: Callable[[EnhancedProductionConfig], None]):
        self.config_file = config_file
        self.callback = callback
        self.last_modified = 0
        self.watching = False
        self.watch_thread = None
        self.logger = logging.getLogger(__name__)
    
    def start_watching(self):
        """Start watching for configuration changes."""
        if not self.watching:
            self.watching = True
            self.watch_thread = threading.Thread(target=self._watch_file)
            self.watch_thread.daemon = True
            self.watch_thread.start()
            self.logger.info("Configuration hot-reloading started")
    
    def stop_watching(self):
        """Stop watching for configuration changes."""
        self.watching = False
        if self.watch_thread:
            self.watch_thread.join()
        self.logger.info("Configuration hot-reloading stopped")
    
    def _watch_file(self):
        """Watch configuration file for changes."""
        while self.watching:
            try:
                if os.path.exists(self.config_file):
                    current_modified = os.path.getmtime(self.config_file)
                    if current_modified > self.last_modified:
                        self.last_modified = current_modified
                        self._reload_config()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error watching config file: {e}")
                time.sleep(5)
    
    def _reload_config(self):
        """Reload configuration and notify callback."""
        try:
            config_manager = EnhancedProductionConfigManager(self.config_file)
            new_config = config_manager.get_config()
            self.callback(new_config)
            self.logger.info("Configuration reloaded successfully")
        except Exception as e:
            self.logger.error(f"Error reloading configuration: {e}")

class EnhancedProductionConfigManager:
    """Enhanced production configuration manager."""
    
    def __init__(self, config_file: Optional[str] = None, environment: Optional[Environment] = None):
        self.config_file = config_file
        self.environment = environment or self._detect_environment()
        self.config = self._load_config()
        self.validator = ConfigValidator()
        self.encryption = None
        self.hot_reloader = None
        self.cache = {}
        self.cache_ttl = {}
        self._setup_logging()
        self._setup_encryption()
        self._setup_hot_reloading()
    
    def _detect_environment(self) -> Environment:
        """Detect environment from environment variables."""
        env = os.getenv("ENVIRONMENT", "development").lower()
        try:
            return Environment(env)
        except ValueError:
            return Environment.DEVELOPMENT
    
    def _load_config(self) -> EnhancedProductionConfig:
        """Load configuration from file or environment."""
        if self.config_file and Path(self.config_file).exists():
            return self._load_from_file()
        else:
            return self._load_from_environment()
    
    def _load_from_file(self) -> EnhancedProductionConfig:
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
    
    def _load_from_environment(self) -> EnhancedProductionConfig:
        """Load configuration from environment variables."""
        config = EnhancedProductionConfig()
        
        # Environment settings
        config.environment = self.environment
        config.debug = os.getenv("DEBUG", "false").lower() == "true"
        config.log_level = LogLevel(os.getenv("LOG_LEVEL", "INFO"))
        
        # Enhanced security settings
        config.security.secret_key = os.getenv("SECRET_KEY", self._generate_secret_key())
        config.security.jwt_secret = os.getenv("JWT_SECRET", self._generate_secret_key())
        config.security.encryption_key = os.getenv("ENCRYPTION_KEY", self._generate_encryption_key())
        
        # Performance settings
        config.performance.max_workers = int(os.getenv("MAX_WORKERS", config.performance.max_workers))
        config.performance.max_memory_gb = float(os.getenv("MAX_MEMORY_GB", config.performance.max_memory_gb))
        config.performance.batch_size = int(os.getenv("BATCH_SIZE", config.performance.batch_size))
        
        # Monitoring settings
        config.monitoring.enable_prometheus = os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true"
        config.monitoring.prometheus_port = int(os.getenv("PROMETHEUS_PORT", config.monitoring.prometheus_port))
        
        # Application settings
        config.host = os.getenv("HOST", config.host)
        config.port = int(os.getenv("PORT", config.port))
        config.workers = int(os.getenv("WORKERS", config.workers))
        
        return config
    
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> EnhancedProductionConfig:
        """Create configuration from dictionary."""
        config = EnhancedProductionConfig()
        
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
        return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(64))
    
    def _generate_encryption_key(self) -> str:
        """Generate encryption key."""
        return Fernet.generate_key().decode()
    
    def _setup_logging(self):
        """Setup enhanced logging configuration."""
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
    
    def _setup_encryption(self):
        """Setup configuration encryption."""
        if self.config.security.enable_encryption and self.config.security.encryption_key:
            self.encryption = ConfigEncryption(self.config.security.encryption_key)
    
    def _setup_hot_reloading(self):
        """Setup configuration hot-reloading."""
        if self.config_file:
            self.hot_reloader = ConfigHotReloader(
                self.config_file,
                self._on_config_changed
            )
            self.hot_reloader.start_watching()
    
    def _on_config_changed(self, new_config: EnhancedProductionConfig):
        """Handle configuration changes."""
        self.config = new_config
        logging.info("Configuration updated via hot-reload")
    
    def get_config(self) -> EnhancedProductionConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        errors = self.validator.validate_config(self.config)
        if errors:
            logging.error(f"Configuration validation failed: {errors}")
            return False
        return True
    
    def save_config(self, filepath: str, encrypt: bool = False):
        """Save configuration to file."""
        config_dict = self._config_to_dict()
        
        if encrypt and self.encryption:
            encrypted_config = self.encryption.encrypt_config(config_dict)
            with open(filepath, 'w') as f:
                f.write(encrypted_config)
        else:
            with open(filepath, 'w') as f:
                if filepath.endswith('.json'):
                    json.dump(config_dict, f, indent=2)
                elif filepath.endswith(('.yml', '.yaml')):
                    yaml.dump(config_dict, f, default_flow_style=False)
    
    def load_encrypted_config(self, filepath: str):
        """Load encrypted configuration."""
        if not self.encryption:
            raise ValueError("Encryption not configured")
        
        with open(filepath, 'r') as f:
            encrypted_config = f.read()
        
        config_dict = self.encryption.decrypt_config(encrypted_config)
        self.config = self._create_config_from_dict(config_dict)
    
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
    
    def get_cached_config(self, key: str, ttl: int = 3600) -> Optional[Any]:
        """Get cached configuration value."""
        if key in self.cache:
            if time.time() - self.cache_ttl.get(key, 0) < ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.cache_ttl[key]
        return None
    
    def set_cached_config(self, key: str, value: Any):
        """Set cached configuration value."""
        self.cache[key] = value
        self.cache_ttl[key] = time.time()
    
    def clear_cache(self):
        """Clear configuration cache."""
        self.cache.clear()
        self.cache_ttl.clear()
    
    def get_config_hash(self) -> str:
        """Get configuration hash for change detection."""
        config_str = json.dumps(self._config_to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def shutdown(self):
        """Shutdown configuration manager."""
        if self.hot_reloader:
            self.hot_reloader.stop_watching()
        self.clear_cache()

def create_enhanced_production_config(config_file: Optional[str] = None, 
                                    environment: Optional[Environment] = None) -> EnhancedProductionConfigManager:
    """Create enhanced production configuration manager."""
    return EnhancedProductionConfigManager(config_file, environment)

def load_enhanced_production_config(config_file: str) -> EnhancedProductionConfig:
    """Load enhanced production configuration from file."""
    manager = EnhancedProductionConfigManager(config_file)
    return manager.get_config()

if __name__ == "__main__":
    # Example usage
    config_manager = create_enhanced_production_config()
    config = config_manager.get_config()
    
    print(f"Environment: {config.environment.value}")
    print(f"Debug: {config.debug}")
    print(f"Log Level: {config.log_level.value}")
    print(f"Security Level: {config.security.enable_encryption}")
    print(f"Performance: {config.performance.max_workers} workers")
    print(f"Monitoring: Prometheus on port {config.monitoring.prometheus_port}")
    print(f"Cache: {'Enabled' if config.cache.enable_redis_cache else 'Disabled'}")
    print(f"Hot Reload: {'Enabled' if config_manager.hot_reloader else 'Disabled'}")
    
    # Test validation
    if config_manager.validate_config():
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration validation failed")

