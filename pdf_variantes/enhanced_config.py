"""Enhanced configuration with environment-based settings."""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///./pdf_variantes.db"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    pool_pre_ping: bool = True


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 20
    socket_timeout: int = 5


@dataclass
class AIConfig:
    """AI processing configuration."""
    enabled: bool = True
    provider: str = "openai"
    model: str = "gpt-4"
    max_tokens: int = 2000
    temperature: float = 0.7
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    cache_ttl: int = 3600
    max_file_size_mb: int = 100
    max_batch_size: int = 10
    enable_compression: bool = True
    enable_caching: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enabled: bool = True
    metrics_endpoint: str = "/metrics"
    health_check_interval: int = 60
    log_level: str = "INFO"
    enable_profiling: bool = False
    max_log_size_mb: int = 100
    log_retention_days: int = 30


@dataclass
class PDFVariantesConfig:
    """Main configuration class."""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "pdf_upload": True,
        "variant_generation": True,
        "topic_extraction": True,
        "brainstorming": True,
        "ai_enhancement": True,
        "collaboration": True,
        "analytics": True,
        "real_time_processing": True
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "database": {
                "url": self.database.url,
                "pool_size": self.database.pool_size,
                "max_overflow": self.database.max_overflow,
                "echo": self.database.echo,
                "pool_pre_ping": self.database.pool_pre_ping
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
                "password": "***" if self.redis.password else None,
                "max_connections": self.redis.max_connections,
                "socket_timeout": self.redis.socket_timeout
            },
            "ai": {
                "enabled": self.ai.enabled,
                "provider": self.ai.provider,
                "model": self.ai.model,
                "max_tokens": self.ai.max_tokens,
                "temperature": self.ai.temperature,
                "api_key": "***" if self.ai.api_key else None,
                "timeout": self.ai.timeout,
                "max_retries": self.ai.max_retries
            },
            "security": {
                "algorithm": self.security.algorithm,
                "access_token_expire_minutes": self.security.access_token_expire_minutes,
                "refresh_token_expire_days": self.security.refresh_token_expire_days,
                "password_min_length": self.security.password_min_length,
                "max_login_attempts": self.security.max_login_attempts,
                "lockout_duration_minutes": self.security.lockout_duration_minutes
            },
            "performance": {
                "max_concurrent_requests": self.performance.max_concurrent_requests,
                "request_timeout": self.performance.request_timeout,
                "cache_ttl": self.performance.cache_ttl,
                "max_file_size_mb": self.performance.max_file_size_mb,
                "max_batch_size": self.performance.max_batch_size,
                "enable_compression": self.performance.enable_compression,
                "enable_caching": self.performance.enable_caching
            },
            "monitoring": {
                "enabled": self.monitoring.enabled,
                "metrics_endpoint": self.monitoring.metrics_endpoint,
                "health_check_interval": self.monitoring.health_check_interval,
                "log_level": self.monitoring.log_level,
                "enable_profiling": self.monitoring.enable_profiling,
                "max_log_size_mb": self.monitoring.max_log_size_mb,
                "log_retention_days": self.monitoring.log_retention_days
            },
            "features": self.features
        }


class ConfigManager:
    """Enhanced configuration manager."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("./config.json")
        self.config: Optional[PDFVariantesConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file or environment."""
        if self.config_path.exists():
            self._load_from_file()
        else:
            self._load_from_environment()
    
    def _load_from_file(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, "r") as f:
                config_data = json.load(f)
            self.config = self._parse_config(config_data)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading config from file: {e}")
            self._load_from_environment()
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        self.config = PDFVariantesConfig(
            environment=Environment(os.getenv("ENVIRONMENT", "development")),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )
        
        # Database config
        self.config.database.url = os.getenv("DATABASE_URL", self.config.database.url)
        self.config.database.pool_size = int(os.getenv("DB_POOL_SIZE", self.config.database.pool_size))
        
        # Redis config
        self.config.redis.host = os.getenv("REDIS_HOST", self.config.redis.host)
        self.config.redis.port = int(os.getenv("REDIS_PORT", self.config.redis.port))
        self.config.redis.password = os.getenv("REDIS_PASSWORD")
        
        # AI config
        self.config.ai.api_key = os.getenv("OPENAI_API_KEY")
        self.config.ai.enabled = os.getenv("AI_ENABLED", "true").lower() == "true"
        
        # Security config
        self.config.security.secret_key = os.getenv("SECRET_KEY", self.config.security.secret_key)
        
        logger.info(f"Configuration loaded from environment for {self.config.environment.value}")
    
    def _parse_config(self, config_data: Dict[str, Any]) -> PDFVariantesConfig:
        """Parse configuration from dictionary."""
        config = PDFVariantesConfig(
            environment=Environment(config_data.get("environment", "development")),
            debug=config_data.get("debug", False)
        )
        
        # Parse sub-configurations
        if "database" in config_data:
            db_config = config_data["database"]
            config.database = DatabaseConfig(
                url=db_config.get("url", config.database.url),
                pool_size=db_config.get("pool_size", config.database.pool_size),
                max_overflow=db_config.get("max_overflow", config.database.max_overflow),
                echo=db_config.get("echo", config.database.echo),
                pool_pre_ping=db_config.get("pool_pre_ping", config.database.pool_pre_ping)
            )
        
        if "redis" in config_data:
            redis_config = config_data["redis"]
            config.redis = RedisConfig(
                host=redis_config.get("host", config.redis.host),
                port=redis_config.get("port", config.redis.port),
                db=redis_config.get("db", config.redis.db),
                password=redis_config.get("password"),
                max_connections=redis_config.get("max_connections", config.redis.max_connections),
                socket_timeout=redis_config.get("socket_timeout", config.redis.socket_timeout)
            )
        
        if "ai" in config_data:
            ai_config = config_data["ai"]
            config.ai = AIConfig(
                enabled=ai_config.get("enabled", config.ai.enabled),
                provider=ai_config.get("provider", config.ai.provider),
                model=ai_config.get("model", config.ai.model),
                max_tokens=ai_config.get("max_tokens", config.ai.max_tokens),
                temperature=ai_config.get("temperature", config.ai.temperature),
                api_key=ai_config.get("api_key"),
                timeout=ai_config.get("timeout", config.ai.timeout),
                max_retries=ai_config.get("max_retries", config.ai.max_retries)
            )
        
        if "security" in config_data:
            security_config = config_data["security"]
            config.security = SecurityConfig(
                secret_key=security_config.get("secret_key", config.security.secret_key),
                algorithm=security_config.get("algorithm", config.security.algorithm),
                access_token_expire_minutes=security_config.get("access_token_expire_minutes", config.security.access_token_expire_minutes),
                refresh_token_expire_days=security_config.get("refresh_token_expire_days", config.security.refresh_token_expire_days),
                password_min_length=security_config.get("password_min_length", config.security.password_min_length),
                max_login_attempts=security_config.get("max_login_attempts", config.security.max_login_attempts),
                lockout_duration_minutes=security_config.get("lockout_duration_minutes", config.security.lockout_duration_minutes)
            )
        
        if "performance" in config_data:
            perf_config = config_data["performance"]
            config.performance = PerformanceConfig(
                max_concurrent_requests=perf_config.get("max_concurrent_requests", config.performance.max_concurrent_requests),
                request_timeout=perf_config.get("request_timeout", config.performance.request_timeout),
                cache_ttl=perf_config.get("cache_ttl", config.performance.cache_ttl),
                max_file_size_mb=perf_config.get("max_file_size_mb", config.performance.max_file_size_mb),
                max_batch_size=perf_config.get("max_batch_size", config.performance.max_batch_size),
                enable_compression=perf_config.get("enable_compression", config.performance.enable_compression),
                enable_caching=perf_config.get("enable_caching", config.performance.enable_caching)
            )
        
        if "monitoring" in config_data:
            monitor_config = config_data["monitoring"]
            config.monitoring = MonitoringConfig(
                enabled=monitor_config.get("enabled", config.monitoring.enabled),
                metrics_endpoint=monitor_config.get("metrics_endpoint", config.monitoring.metrics_endpoint),
                health_check_interval=monitor_config.get("health_check_interval", config.monitoring.health_check_interval),
                log_level=monitor_config.get("log_level", config.monitoring.log_level),
                enable_profiling=monitor_config.get("enable_profiling", config.monitoring.enable_profiling),
                max_log_size_mb=monitor_config.get("max_log_size_mb", config.monitoring.max_log_size_mb),
                log_retention_days=monitor_config.get("log_retention_days", config.monitoring.log_retention_days)
            )
        
        if "features" in config_data:
            config.features.update(config_data["features"])
        
        return config
    
    def get_config(self) -> PDFVariantesConfig:
        """Get current configuration."""
        if not self.config:
            self._load_config()
        return self.config
    
    def save_config(self, config: Optional[PDFVariantesConfig] = None) -> None:
        """Save configuration to file."""
        config_to_save = config or self.config
        if not config_to_save:
            raise ValueError("No configuration to save")
        
        config_dict = config_to_save.to_dict()
        
        with open(self.config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {self.config_path}")
    
    def update_feature(self, feature_name: str, enabled: bool) -> None:
        """Update feature toggle."""
        if not self.config:
            self._load_config()
        
        self.config.features[feature_name] = enabled
        self.save_config()
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if feature is enabled."""
        if not self.config:
            self._load_config()
        
        return self.config.features.get(feature_name, False)
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        if not self.config:
            self._load_config()
        
        env_config = {
            "development": {
                "debug": True,
                "log_level": "DEBUG",
                "enable_profiling": True,
                "cache_ttl": 60
            },
            "staging": {
                "debug": False,
                "log_level": "INFO",
                "enable_profiling": False,
                "cache_ttl": 300
            },
            "production": {
                "debug": False,
                "log_level": "WARNING",
                "enable_profiling": False,
                "cache_ttl": 3600
            },
            "testing": {
                "debug": True,
                "log_level": "DEBUG",
                "enable_profiling": False,
                "cache_ttl": 0
            }
        }
        
        return env_config.get(self.config.environment.value, {})
