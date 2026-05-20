"""
Unified Configuration Management System

Provides a centralized, type-safe configuration system for the entire
email sequence system with environment-based overrides and validation.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import yaml
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str = "email_sequence"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    enable_caching: bool = True
    cache_ttl: int = 3600
    max_cache_size: int = 10000
    enable_metrics: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0

    @property
    def connection_string(self) -> str:
        """Generate connection string"""
        if self.password:
            return f"{self.type}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"{self.type}://{self.username}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: str = ""
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 2.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30

    @property
    def connection_string(self) -> str:
        """Generate Redis connection string"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"redis://{self.host}:{self.port}/{self.database}"


@dataclass
class QueueConfig:
    """Message queue configuration"""
    type: str = "redis_streams"  # redis_streams, rabbitmq, kafka, memory
    host: str = "localhost"
    port: int = 6379
    username: str = ""
    password: str = ""
    max_queue_size: int = 10000
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100
    enable_compression: bool = True
    enable_encryption: bool = False
    enable_metrics: bool = True
    enable_dlq: bool = True
    dlq_max_retries: int = 3
    stream_max_len: int = 1000

    @property
    def connection_string(self) -> str:
        """Generate queue connection string"""
        if self.type == "redis_streams":
            if self.password:
                return f"redis://:{self.password}@{self.host}:{self.port}"
            return f"redis://{self.host}:{self.port}"
        elif self.type == "rabbitmq":
            if self.username and self.password:
                return f"amqp://{self.username}:{self.password}@{self.host}:{self.port}/"
            return f"amqp://{self.host}:{self.port}/"
        elif self.type == "kafka":
            return f"{self.host}:{self.port}"
        return "memory"


@dataclass
class SecurityConfig:
    """Security configuration"""
    security_level: str = "medium"  # low, medium, high, critical
    enable_encryption: bool = True
    enable_rate_limiting: bool = True
    enable_audit_logging: bool = True
    enable_jwt_auth: bool = True
    enable_bcrypt_hashing: bool = True
    encryption_type: str = "symmetric"  # symmetric, asymmetric, hybrid
    jwt_secret: str = ""
    encryption_key: str = ""
    rate_limit_requests_per_minute: int = 100
    rate_limit_requests_per_hour: int = 1000
    rate_limit_requests_per_day: int = 10000
    audit_log_retention_days: int = 90
    max_password_length: int = 128
    min_password_length: int = 8
    password_complexity_required: bool = True
    session_timeout_minutes: int = 60
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    max_memory_usage: float = 0.8
    cache_size: int = 1000
    batch_size: int = 64
    max_concurrent_tasks: int = 10
    enable_caching: bool = True
    enable_memory_optimization: bool = True
    enable_batch_processing: bool = True
    enable_ml_optimization: bool = True
    enable_predictive_caching: bool = True
    enable_adaptive_batching: bool = True
    enable_intelligent_resource_management: bool = True
    enable_performance_prediction: bool = True
    ml_model_path: str = "models/optimization_model.pkl"
    max_sequence_length: int = 512
    enable_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 4
    enable_mixed_precision: bool = True
    enable_multi_gpu: bool = False
    gpu_memory_fraction: float = 0.8


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    monitoring_interval: int = 5
    alert_threshold: float = 0.8
    auto_optimization_enabled: bool = True
    enable_real_time_alerts: bool = True
    enable_performance_tracking: bool = True
    enable_resource_monitoring: bool = True
    enable_ml_insights: bool = True
    enable_anomaly_detection: bool = True
    enable_health_checks: bool = True
    health_check_interval: int = 30
    metrics_retention_days: int = 30
    alert_channels: List[str] = field(default_factory=lambda: ["console", "email"])
    alert_email_recipients: List[str] = field(default_factory=list)
    enable_prometheus_metrics: bool = True
    prometheus_port: int = 9090


@dataclass
class StreamingConfig:
    """Real-time streaming configuration"""
    stream_type: str = "websocket"  # websocket, sse, http
    enable_real_time: bool = True
    max_connections: int = 1000
    connection_timeout: int = 300
    heartbeat_interval: int = 30
    enable_compression: bool = True
    enable_authentication: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_rate_limiting: bool = True
    max_messages_per_second: int = 100
    enable_event_persistence: bool = False
    event_retention_hours: int = 24


@dataclass
class EmailConfig:
    """Email service configuration"""
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    from_email: str = "noreply@example.com"
    from_name: str = "Email Sequence System"
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100
    enable_tracking: bool = True
    enable_analytics: bool = True
    template_path: str = "templates"
    enable_dkim: bool = False
    dkim_private_key: str = ""
    dkim_selector: str = "default"


@dataclass
class LangChainConfig:
    """LangChain service configuration"""
    model_name: str = "gpt-3.5-turbo"
    api_key: str = ""
    api_base: str = ""
    max_tokens: int = 1000
    temperature: float = 0.7
    enable_streaming: bool = True
    enable_caching: bool = True
    cache_ttl: int = 3600
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    enable_fallback: bool = True
    fallback_model: str = "gpt-3.5-turbo"
    enable_logging: bool = True
    log_level: str = "info"


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/email_sequence.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True
    enable_json: bool = False
    enable_structured_logging: bool = True
    log_sensitive_data: bool = False
    enable_performance_logging: bool = True
    performance_log_interval: int = 60


class UnifiedConfig:
    """
    Unified configuration management system for the email sequence system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.environment = self._get_environment()
        
        # Initialize all configuration sections
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.queue = QueueConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        self.monitoring = MonitoringConfig()
        self.streaming = StreamingConfig()
        self.email = EmailConfig()
        self.langchain = LangChainConfig()
        self.logging = LoggingConfig()
        
        # Load configuration
        self._load_configuration()
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(f"Configuration loaded for environment: {self.environment.value}")
    
    def _get_default_config_path(self) -> str:
        """Get default configuration path"""
        return os.getenv("EMAIL_SEQUENCE_CONFIG_PATH", "config/email_sequence.yaml")
    
    def _get_environment(self) -> Environment:
        """Get current environment"""
        env_str = os.getenv("EMAIL_SEQUENCE_ENV", "development").lower()
        try:
            return Environment(env_str)
        except ValueError:
            logger.warning(f"Invalid environment '{env_str}', using development")
            return Environment.DEVELOPMENT
    
    def _load_configuration(self) -> None:
        """Load configuration from file and environment variables"""
        try:
            # Load from file if exists
            if os.path.exists(self.config_path):
                self._load_from_file()
            
            # Override with environment variables
            self._load_from_environment()
            
            # Set environment-specific defaults
            self._set_environment_defaults()
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _load_from_file(self) -> None:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.json'):
                    config_data = json.load(f)
                elif self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
            
            # Apply configuration to sections
            self._apply_config_data(config_data)
            
        except Exception as e:
            logger.error(f"Error loading config file {self.config_path}: {e}")
            raise
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables"""
        env_prefix = "EMAIL_SEQUENCE_"
        
        # Database configuration
        if os.getenv(f"{env_prefix}DB_HOST"):
            self.database.host = os.getenv(f"{env_prefix}DB_HOST")
        if os.getenv(f"{env_prefix}DB_PORT"):
            self.database.port = int(os.getenv(f"{env_prefix}DB_PORT"))
        if os.getenv(f"{env_prefix}DB_NAME"):
            self.database.database = os.getenv(f"{env_prefix}DB_NAME")
        if os.getenv(f"{env_prefix}DB_USER"):
            self.database.username = os.getenv(f"{env_prefix}DB_USER")
        if os.getenv(f"{env_prefix}DB_PASS"):
            self.database.password = os.getenv(f"{env_prefix}DB_PASS")
        
        # Redis configuration
        if os.getenv(f"{env_prefix}REDIS_HOST"):
            self.redis.host = os.getenv(f"{env_prefix}REDIS_HOST")
        if os.getenv(f"{env_prefix}REDIS_PORT"):
            self.redis.port = int(os.getenv(f"{env_prefix}REDIS_PORT"))
        if os.getenv(f"{env_prefix}REDIS_PASS"):
            self.redis.password = os.getenv(f"{env_prefix}REDIS_PASS")
        
        # Security configuration
        if os.getenv(f"{env_prefix}JWT_SECRET"):
            self.security.jwt_secret = os.getenv(f"{env_prefix}JWT_SECRET")
        if os.getenv(f"{env_prefix}ENCRYPTION_KEY"):
            self.security.encryption_key = os.getenv(f"{env_prefix}ENCRYPTION_KEY")
        
        # LangChain configuration
        if os.getenv(f"{env_prefix}LANGCHAIN_API_KEY"):
            self.langchain.api_key = os.getenv(f"{env_prefix}LANGCHAIN_API_KEY")
        if os.getenv(f"{env_prefix}LANGCHAIN_MODEL"):
            self.langchain.model_name = os.getenv(f"{env_prefix}LANGCHAIN_MODEL")
        
        # Email configuration
        if os.getenv(f"{env_prefix}SMTP_HOST"):
            self.email.smtp_host = os.getenv(f"{env_prefix}SMTP_HOST")
        if os.getenv(f"{env_prefix}SMTP_PORT"):
            self.email.smtp_port = int(os.getenv(f"{env_prefix}SMTP_PORT"))
        if os.getenv(f"{env_prefix}SMTP_USER"):
            self.email.smtp_username = os.getenv(f"{env_prefix}SMTP_USER")
        if os.getenv(f"{env_prefix}SMTP_PASS"):
            self.email.smtp_password = os.getenv(f"{env_prefix}SMTP_PASS")
    
    def _set_environment_defaults(self) -> None:
        """Set environment-specific configuration defaults"""
        if self.environment == Environment.DEVELOPMENT:
            self.logging.level = LogLevel.DEBUG
            self.monitoring.monitoring_interval = 2
            self.performance.enable_ml_optimization = False
            self.security.security_level = "low"
            
        elif self.environment == Environment.STAGING:
            self.logging.level = LogLevel.INFO
            self.monitoring.monitoring_interval = 5
            self.performance.enable_ml_optimization = True
            self.security.security_level = "medium"
            
        elif self.environment == Environment.PRODUCTION:
            self.logging.level = LogLevel.WARNING
            self.monitoring.monitoring_interval = 10
            self.performance.enable_ml_optimization = True
            self.security.security_level = "high"
            self.monitoring.auto_optimization_enabled = True
            
        elif self.environment == Environment.TESTING:
            self.logging.level = LogLevel.DEBUG
            self.monitoring.monitoring_interval = 1
            self.performance.enable_ml_optimization = False
            self.security.security_level = "low"
            self.database.database = "email_sequence_test"
    
    def _apply_config_data(self, config_data: Dict[str, Any]) -> None:
        """Apply configuration data to sections"""
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def _validate_configuration(self) -> None:
        """Validate configuration settings"""
        errors = []
        
        # Validate database configuration
        if not self.database.host:
            errors.append("Database host is required")
        
        # Validate Redis configuration
        if not self.redis.host:
            errors.append("Redis host is required")
        
        # Validate security configuration
        if self.security.enable_jwt_auth and not self.security.jwt_secret:
            errors.append("JWT secret is required when JWT auth is enabled")
        
        if self.security.enable_encryption and not self.security.encryption_key:
            errors.append("Encryption key is required when encryption is enabled")
        
        # Validate LangChain configuration
        if not self.langchain.api_key:
            errors.append("LangChain API key is required")
        
        # Validate email configuration
        if not self.email.smtp_host:
            errors.append("SMTP host is required")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            "environment": self.environment.value,
            "database": self.database.__dict__,
            "redis": self.redis.__dict__,
            "queue": self.queue.__dict__,
            "security": self.security.__dict__,
            "performance": self.performance.__dict__,
            "monitoring": self.monitoring.__dict__,
            "streaming": self.streaming.__dict__,
            "email": self.email.__dict__,
            "langchain": self.langchain.__dict__,
            "logging": self.logging.__dict__
        }
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file"""
        save_path = path or self.config_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        config_data = self.get_config_dict()
        
        with open(save_path, 'w') as f:
            if save_path.endswith('.json'):
                json.dump(config_data, f, indent=2)
            elif save_path.endswith('.yaml') or save_path.endswith('.yml'):
                yaml.dump(config_data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config file format: {save_path}")
        
        logger.info(f"Configuration saved to {save_path}")
    
    def get_section(self, section_name: str) -> Any:
        """Get configuration section by name"""
        if hasattr(self, section_name):
            return getattr(self, section_name)
        raise ValueError(f"Configuration section '{section_name}' not found")
    
    def update_section(self, section_name: str, **kwargs) -> None:
        """Update configuration section"""
        if hasattr(self, section_name):
            section = getattr(self, section_name)
            for key, value in kwargs.items():
                if hasattr(section, key):
                    setattr(section, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")
        else:
            raise ValueError(f"Configuration section '{section_name}' not found")
    
    def reload(self) -> None:
        """Reload configuration from file and environment"""
        self._load_configuration()
        logger.info("Configuration reloaded") 