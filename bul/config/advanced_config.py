"""
Advanced Configuration System - Functional Programming Approach
============================================================

Implementation of advanced configuration management following functional programming principles:
- Immutable configuration
- Environment-based configuration
- Validation and type safety
- Hot reloading capabilities
- Configuration composition
"""

from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic
from functools import partial, lru_cache
from dataclasses import dataclass, field
from enum import Enum
import os
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import PositiveInt, NonNegativeInt

# Type variables
T = TypeVar('T')

# Configuration Environment
class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class CacheBackend(str, Enum):
    """Cache backend types"""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"

# Advanced Configuration Models
@dataclass(frozen=True)
class DatabaseConfig:
    """Immutable database configuration"""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    
    @classmethod
    def from_env(cls, prefix: str = "DATABASE_") -> 'DatabaseConfig':
        """Create database config from environment variables"""
        return cls(
            url=os.getenv(f"{prefix}URL", "sqlite:///bul.db"),
            pool_size=int(os.getenv(f"{prefix}POOL_SIZE", "10")),
            max_overflow=int(os.getenv(f"{prefix}MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv(f"{prefix}POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv(f"{prefix}POOL_RECYCLE", "3600")),
            echo=os.getenv(f"{prefix}ECHO", "false").lower() == "true"
        )

@dataclass(frozen=True)
class CacheConfig:
    """Immutable cache configuration"""
    backend: CacheBackend
    url: Optional[str] = None
    max_size: int = 1000
    default_ttl: int = 3600
    enabled: bool = True
    
    @classmethod
    def from_env(cls, prefix: str = "CACHE_") -> 'CacheConfig':
        """Create cache config from environment variables"""
        return cls(
            backend=CacheBackend(os.getenv(f"{prefix}BACKEND", "memory")),
            url=os.getenv(f"{prefix}URL"),
            max_size=int(os.getenv(f"{prefix}MAX_SIZE", "1000")),
            default_ttl=int(os.getenv(f"{prefix}DEFAULT_TTL", "3600")),
            enabled=os.getenv(f"{prefix}ENABLED", "true").lower() == "true"
        )

@dataclass(frozen=True)
class SecurityConfig:
    """Immutable security configuration"""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    @classmethod
    def from_env(cls, prefix: str = "SECURITY_") -> 'SecurityConfig':
        """Create security config from environment variables"""
        return cls(
            secret_key=os.getenv(f"{prefix}SECRET_KEY", "your-secret-key-here"),
            algorithm=os.getenv(f"{prefix}ALGORITHM", "HS256"),
            access_token_expire_minutes=int(os.getenv(f"{prefix}ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
            refresh_token_expire_days=int(os.getenv(f"{prefix}REFRESH_TOKEN_EXPIRE_DAYS", "7")),
            password_min_length=int(os.getenv(f"{prefix}PASSWORD_MIN_LENGTH", "8")),
            max_failed_attempts=int(os.getenv(f"{prefix}MAX_FAILED_ATTEMPTS", "5")),
            lockout_duration_minutes=int(os.getenv(f"{prefix}LOCKOUT_DURATION_MINUTES", "15")),
            rate_limit_requests=int(os.getenv(f"{prefix}RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.getenv(f"{prefix}RATE_LIMIT_WINDOW", "60"))
        )

@dataclass(frozen=True)
class APIConfig:
    """Immutable API configuration"""
    openrouter_api_key: str
    openai_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openai_base_url: str = "https://api.openai.com/v1"
    default_model: str = "openai/gpt-4"
    fallback_model: str = "openai/gpt-3.5-turbo"
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3
    
    @classmethod
    def from_env(cls, prefix: str = "API_") -> 'APIConfig':
        """Create API config from environment variables"""
        return cls(
            openrouter_api_key=os.getenv(f"{prefix}OPENROUTER_API_KEY", ""),
            openai_api_key=os.getenv(f"{prefix}OPENAI_API_KEY"),
            openrouter_base_url=os.getenv(f"{prefix}OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openai_base_url=os.getenv(f"{prefix}OPENAI_BASE_URL", "https://api.openai.com/v1"),
            default_model=os.getenv(f"{prefix}DEFAULT_MODEL", "openai/gpt-4"),
            fallback_model=os.getenv(f"{prefix}FALLBACK_MODEL", "openai/gpt-3.5-turbo"),
            max_tokens=int(os.getenv(f"{prefix}MAX_TOKENS", "4000")),
            temperature=float(os.getenv(f"{prefix}TEMPERATURE", "0.7")),
            timeout=int(os.getenv(f"{prefix}TIMEOUT", "30")),
            max_retries=int(os.getenv(f"{prefix}MAX_RETRIES", "3"))
        )

@dataclass(frozen=True)
class ServerConfig:
    """Immutable server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: LogLevel = LogLevel.INFO
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    enable_compression: bool = True
    
    @classmethod
    def from_env(cls, prefix: str = "SERVER_") -> 'ServerConfig':
        """Create server config from environment variables"""
        cors_origins = os.getenv(f"{prefix}CORS_ORIGINS", "*")
        if cors_origins == "*":
            cors_origins = ["*"]
        else:
            cors_origins = [origin.strip() for origin in cors_origins.split(",")]
        
        return cls(
            host=os.getenv(f"{prefix}HOST", "0.0.0.0"),
            port=int(os.getenv(f"{prefix}PORT", "8000")),
            workers=int(os.getenv(f"{prefix}WORKERS", "1")),
            reload=os.getenv(f"{prefix}RELOAD", "false").lower() == "true",
            log_level=LogLevel(os.getenv(f"{prefix}LOG_LEVEL", "INFO")),
            cors_origins=cors_origins,
            max_request_size=int(os.getenv(f"{prefix}MAX_REQUEST_SIZE", str(16 * 1024 * 1024))),
            enable_compression=os.getenv(f"{prefix}ENABLE_COMPRESSION", "true").lower() == "true"
        )

@dataclass(frozen=True)
class LoggingConfig:
    """Immutable logging configuration"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    json_logs: bool = False
    
    @classmethod
    def from_env(cls, prefix: str = "LOGGING_") -> 'LoggingConfig':
        """Create logging config from environment variables"""
        return cls(
            level=LogLevel(os.getenv(f"{prefix}LEVEL", "INFO")),
            format=os.getenv(f"{prefix}FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file_path=os.getenv(f"{prefix}FILE_PATH"),
            max_file_size=int(os.getenv(f"{prefix}MAX_FILE_SIZE", str(10 * 1024 * 1024))),
            backup_count=int(os.getenv(f"{prefix}BACKUP_COUNT", "5")),
            json_logs=os.getenv(f"{prefix}JSON_LOGS", "false").lower() == "true"
        )

# Main Configuration Class
@dataclass(frozen=True)
class BULConfig:
    """Immutable main configuration"""
    environment: Environment
    debug: bool = False
    
    # Sub-configurations
    database: DatabaseConfig
    cache: CacheConfig
    security: SecurityConfig
    api: APIConfig
    server: ServerConfig
    logging: LoggingConfig
    
    # Business logic configuration
    max_documents_per_batch: int = 10
    max_document_length: int = 50000
    default_language: str = "es"
    supported_languages: List[str] = field(default_factory=lambda: ["es", "en", "pt", "fr"])
    default_format: str = "markdown"
    supported_formats: List[str] = field(default_factory=lambda: ["markdown", "html", "pdf", "docx"])
    quality_threshold: float = 0.7
    max_processing_time: int = 300
    
    @classmethod
    def from_env(cls) -> 'BULConfig':
        """Create configuration from environment variables"""
        environment = Environment(os.getenv("ENVIRONMENT", "development"))
        debug = os.getenv("DEBUG", "false").lower() == "true"
        
        return cls(
            environment=environment,
            debug=debug,
            database=DatabaseConfig.from_env(),
            cache=CacheConfig.from_env(),
            security=SecurityConfig.from_env(),
            api=APIConfig.from_env(),
            server=ServerConfig.from_env(),
            logging=LoggingConfig.from_env(),
            max_documents_per_batch=int(os.getenv("MAX_DOCUMENTS_PER_BATCH", "10")),
            max_document_length=int(os.getenv("MAX_DOCUMENT_LENGTH", "50000")),
            default_language=os.getenv("DEFAULT_LANGUAGE", "es"),
            supported_languages=os.getenv("SUPPORTED_LANGUAGES", "es,en,pt,fr").split(","),
            default_format=os.getenv("DEFAULT_FORMAT", "markdown"),
            supported_formats=os.getenv("SUPPORTED_FORMATS", "markdown,html,pdf,docx").split(","),
            quality_threshold=float(os.getenv("QUALITY_THRESHOLD", "0.7")),
            max_processing_time=int(os.getenv("MAX_PROCESSING_TIME", "300"))
        )
    
    @classmethod
    def from_file(cls, file_path: str) -> 'BULConfig':
        """Create configuration from file"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.suffix.lower() in ['.yml', '.yaml']:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")
        
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'BULConfig':
        """Create configuration from dictionary"""
        return cls(
            environment=Environment(data.get("environment", "development")),
            debug=data.get("debug", False),
            database=DatabaseConfig(**data.get("database", {})),
            cache=CacheConfig(**data.get("cache", {})),
            security=SecurityConfig(**data.get("security", {})),
            api=APIConfig(**data.get("api", {})),
            server=ServerConfig(**data.get("server", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            max_documents_per_batch=data.get("max_documents_per_batch", 10),
            max_document_length=data.get("max_document_length", 50000),
            default_language=data.get("default_language", "es"),
            supported_languages=data.get("supported_languages", ["es", "en", "pt", "fr"]),
            default_format=data.get("default_format", "markdown"),
            supported_formats=data.get("supported_formats", ["markdown", "html", "pdf", "docx"]),
            quality_threshold=data.get("quality_threshold", 0.7),
            max_processing_time=data.get("max_processing_time", 300)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "database": {
                "url": self.database.url,
                "pool_size": self.database.pool_size,
                "max_overflow": self.database.max_overflow,
                "pool_timeout": self.database.pool_timeout,
                "pool_recycle": self.database.pool_recycle,
                "echo": self.database.echo
            },
            "cache": {
                "backend": self.cache.backend.value,
                "url": self.cache.url,
                "max_size": self.cache.max_size,
                "default_ttl": self.cache.default_ttl,
                "enabled": self.cache.enabled
            },
            "security": {
                "secret_key": "***",  # Hide sensitive data
                "algorithm": self.security.algorithm,
                "access_token_expire_minutes": self.security.access_token_expire_minutes,
                "refresh_token_expire_days": self.security.refresh_token_expire_days,
                "password_min_length": self.security.password_min_length,
                "max_failed_attempts": self.security.max_failed_attempts,
                "lockout_duration_minutes": self.security.lockout_duration_minutes,
                "rate_limit_requests": self.security.rate_limit_requests,
                "rate_limit_window": self.security.rate_limit_window
            },
            "api": {
                "openrouter_api_key": "***",  # Hide sensitive data
                "openai_api_key": "***" if self.api.openai_api_key else None,
                "openrouter_base_url": self.api.openrouter_base_url,
                "openai_base_url": self.api.openai_base_url,
                "default_model": self.api.default_model,
                "fallback_model": self.api.fallback_model,
                "max_tokens": self.api.max_tokens,
                "temperature": self.api.temperature,
                "timeout": self.api.timeout,
                "max_retries": self.api.max_retries
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers,
                "reload": self.server.reload,
                "log_level": self.server.log_level.value,
                "cors_origins": self.server.cors_origins,
                "max_request_size": self.server.max_request_size,
                "enable_compression": self.server.enable_compression
            },
            "logging": {
                "level": self.logging.level.value,
                "format": self.logging.format,
                "file_path": self.logging.file_path,
                "max_file_size": self.logging.max_file_size,
                "backup_count": self.logging.backup_count,
                "json_logs": self.logging.json_logs
            },
            "max_documents_per_batch": self.max_documents_per_batch,
            "max_document_length": self.max_document_length,
            "default_language": self.default_language,
            "supported_languages": self.supported_languages,
            "default_format": self.default_format,
            "supported_formats": self.supported_formats,
            "quality_threshold": self.quality_threshold,
            "max_processing_time": self.max_processing_time
        }
    
    def validate(self) -> bool:
        """Validate configuration"""
        try:
            # Validate required fields
            if not self.api.openrouter_api_key:
                raise ValueError("OpenRouter API key is required")
            
            if not self.security.secret_key or len(self.security.secret_key) < 32:
                raise ValueError("Secret key must be at least 32 characters long")
            
            # Validate environment-specific settings
            if self.environment == Environment.PRODUCTION:
                if self.debug:
                    raise ValueError("Debug mode cannot be enabled in production")
                
                if "*" in self.server.cors_origins:
                    raise ValueError("CORS cannot allow all origins in production")
            
            # Validate database URL
            if not self.database.url:
                raise ValueError("Database URL is required")
            
            # Validate cache configuration
            if self.cache.backend == CacheBackend.REDIS and not self.cache.url:
                raise ValueError("Redis URL is required when using Redis backend")
            
            return True
            
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            return False

# Advanced Configuration Manager
class ConfigurationManager:
    """Advanced configuration manager with hot reloading"""
    
    def __init__(self, config: BULConfig):
        self._config = config
        self._watchers: List[Callable[[BULConfig], None]] = []
        self._file_path: Optional[str] = None
        self._last_modified: Optional[float] = None
    
    @property
    def config(self) -> BULConfig:
        """Get current configuration"""
        return self._config
    
    def add_watcher(self, watcher: Callable[[BULConfig], None]) -> None:
        """Add configuration change watcher"""
        self._watchers.append(watcher)
    
    def remove_watcher(self, watcher: Callable[[BULConfig], None]) -> None:
        """Remove configuration change watcher"""
        if watcher in self._watchers:
            self._watchers.remove(watcher)
    
    def reload_from_file(self, file_path: str) -> bool:
        """Reload configuration from file"""
        try:
            new_config = BULConfig.from_file(file_path)
            
            if new_config.validate():
                self._config = new_config
                self._file_path = file_path
                self._last_modified = os.path.getmtime(file_path)
                
                # Notify watchers
                for watcher in self._watchers:
                    try:
                        watcher(new_config)
                    except Exception as e:
                        logging.error(f"Error in configuration watcher: {e}")
                
                return True
            else:
                logging.error("Configuration validation failed")
                return False
                
        except Exception as e:
            logging.error(f"Failed to reload configuration: {e}")
            return False
    
    def reload_from_env(self) -> bool:
        """Reload configuration from environment"""
        try:
            new_config = BULConfig.from_env()
            
            if new_config.validate():
                self._config = new_config
                
                # Notify watchers
                for watcher in self._watchers:
                    try:
                        watcher(new_config)
                    except Exception as e:
                        logging.error(f"Error in configuration watcher: {e}")
                
                return True
            else:
                logging.error("Configuration validation failed")
                return False
                
        except Exception as e:
            logging.error(f"Failed to reload configuration: {e}")
            return False
    
    def check_file_changes(self) -> bool:
        """Check if configuration file has changed"""
        if not self._file_path or not os.path.exists(self._file_path):
            return False
        
        current_modified = os.path.getmtime(self._file_path)
        
        if self._last_modified is None or current_modified > self._last_modified:
            return self.reload_from_file(self._file_path)
        
        return False
    
    def get_config_value(self, path: str, default: Any = None) -> Any:
        """Get configuration value by path (e.g., 'database.url')"""
        keys = path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = getattr(value, key)
            return value
        except AttributeError:
            return default
    
    def set_config_value(self, path: str, value: Any) -> bool:
        """Set configuration value by path (creates new config)"""
        # This would require creating a new configuration object
        # since the configuration is immutable
        logging.warning("Configuration is immutable. Cannot set values directly.")
        return False

# Global configuration manager
_config_manager: Optional[ConfigurationManager] = None

def get_config() -> BULConfig:
    """Get global configuration"""
    global _config_manager
    if _config_manager is None:
        config = BULConfig.from_env()
        _config_manager = ConfigurationManager(config)
    return _config_manager.config

def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager"""
    global _config_manager
    if _config_manager is None:
        config = BULConfig.from_env()
        _config_manager = ConfigurationManager(config)
    return _config_manager

def reload_config() -> bool:
    """Reload configuration from environment"""
    return get_config_manager().reload_from_env()

def reload_config_from_file(file_path: str) -> bool:
    """Reload configuration from file"""
    return get_config_manager().reload_from_file(file_path)

def validate_config() -> bool:
    """Validate current configuration"""
    return get_config().validate()

# Environment-specific configuration
def is_production() -> bool:
    """Check if running in production"""
    return get_config().environment == Environment.PRODUCTION

def is_development() -> bool:
    """Check if running in development"""
    return get_config().environment == Environment.DEVELOPMENT

def is_staging() -> bool:
    """Check if running in staging"""
    return get_config().environment == Environment.STAGING

def is_testing() -> bool:
    """Check if running in testing"""
    return get_config().environment == Environment.TESTING

# Configuration utilities
@lru_cache(maxsize=128)
def get_cached_config_value(path: str, default: Any = None) -> Any:
    """Get cached configuration value"""
    return get_config_manager().get_config_value(path, default)

def create_config_validator(required_fields: List[str]) -> Callable[[BULConfig], bool]:
    """Create configuration validator for required fields"""
    def validator(config: BULConfig) -> bool:
        for field in required_fields:
            if not getattr(config, field, None):
                logging.error(f"Required configuration field missing: {field}")
                return False
        return True
    return validator

def create_environment_validator(environment: Environment) -> Callable[[BULConfig], bool]:
    """Create environment-specific validator"""
    def validator(config: BULConfig) -> bool:
        if config.environment != environment:
            logging.error(f"Configuration environment mismatch. Expected: {environment}, Got: {config.environment}")
            return False
        return True
    return validator

# Export all configuration utilities
__all__ = [
    # Enums
    "Environment",
    "LogLevel", 
    "CacheBackend",
    
    # Configuration Classes
    "DatabaseConfig",
    "CacheConfig",
    "SecurityConfig",
    "APIConfig",
    "ServerConfig",
    "LoggingConfig",
    "BULConfig",
    
    # Configuration Manager
    "ConfigurationManager",
    
    # Global Functions
    "get_config",
    "get_config_manager",
    "reload_config",
    "reload_config_from_file",
    "validate_config",
    
    # Environment Checks
    "is_production",
    "is_development",
    "is_staging",
    "is_testing",
    
    # Utilities
    "get_cached_config_value",
    "create_config_validator",
    "create_environment_validator"
]