"""
Configuration Manager - Centralized Configuration
===============================================

Centralized configuration management with validation and environment support.
"""

import os
import json
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    backend: str = "memory"  # memory, redis, disk
    redis_url: Optional[str] = None
    max_memory_mb: int = 1024
    default_ttl: int = 3600
    compression: bool = True


@dataclass
class AIConfig:
    """AI service configuration."""
    provider: str = "openai"  # openai, anthropic, cohere, local
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 2000
    temperature: float = 0.7
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class ProcessingConfig:
    """Document processing configuration."""
    max_file_size_mb: int = 100
    allowed_extensions: List[str] = field(default_factory=lambda: [".md", ".pdf", ".docx", ".doc", ".txt"])
    max_workers: int = 8
    chunk_size: int = 8192
    enable_streaming: bool = True
    enable_parallel: bool = True
    temp_dir: str = "/tmp"


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    enable_authentication: bool = False
    secret_key: Optional[str] = None
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = False
    jaeger_endpoint: Optional[str] = None


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8001
    workers: int = 1
    reload: bool = False
    access_log: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (JSON or YAML)
        """
        self.config_file = Path(config_file) if config_file else None
        self._configs = {}
        self._load_configurations()
    
    def _load_configurations(self):
        """Load configurations from environment and config file."""
        # Load from environment variables
        self._load_from_env()
        
        # Load from config file if provided
        if self.config_file and self.config_file.exists():
            self._load_from_file()
        
        # Validate configurations
        self._validate_configs()
    
    def _load_from_env(self):
        """Load configurations from environment variables."""
        # Database config
        self._configs['database'] = DatabaseConfig(
            url=os.getenv('DATABASE_URL'),
            pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '20')),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30')),
            pool_recycle=int(os.getenv('DB_POOL_RECYCLE', '3600'))
        )
        
        # Cache config
        self._configs['cache'] = CacheConfig(
            enabled=os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
            backend=os.getenv('CACHE_BACKEND', 'memory'),
            redis_url=os.getenv('REDIS_URL'),
            max_memory_mb=int(os.getenv('CACHE_MAX_MEMORY_MB', '1024')),
            default_ttl=int(os.getenv('CACHE_DEFAULT_TTL', '3600')),
            compression=os.getenv('CACHE_COMPRESSION', 'true').lower() == 'true'
        )
        
        # AI config
        self._configs['ai'] = AIConfig(
            provider=os.getenv('AI_PROVIDER', 'openai'),
            api_key=os.getenv('AI_API_KEY'),
            model=os.getenv('AI_MODEL', 'gpt-3.5-turbo'),
            max_tokens=int(os.getenv('AI_MAX_TOKENS', '2000')),
            temperature=float(os.getenv('AI_TEMPERATURE', '0.7')),
            timeout=int(os.getenv('AI_TIMEOUT', '30')),
            retry_attempts=int(os.getenv('AI_RETRY_ATTEMPTS', '3')),
            retry_delay=float(os.getenv('AI_RETRY_DELAY', '1.0'))
        )
        
        # Processing config
        allowed_extensions = os.getenv('ALLOWED_EXTENSIONS', '.md,.pdf,.docx,.doc,.txt')
        self._configs['processing'] = ProcessingConfig(
            max_file_size_mb=int(os.getenv('MAX_FILE_SIZE_MB', '100')),
            allowed_extensions=allowed_extensions.split(','),
            max_workers=int(os.getenv('MAX_WORKERS', '8')),
            chunk_size=int(os.getenv('CHUNK_SIZE', '8192')),
            enable_streaming=os.getenv('ENABLE_STREAMING', 'true').lower() == 'true',
            enable_parallel=os.getenv('ENABLE_PARALLEL', 'true').lower() == 'true',
            temp_dir=os.getenv('TEMP_DIR', '/tmp')
        )
        
        # Security config
        allowed_origins = os.getenv('ALLOWED_ORIGINS', '*')
        self._configs['security'] = SecurityConfig(
            enable_rate_limiting=os.getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true',
            rate_limit_requests=int(os.getenv('RATE_LIMIT_REQUESTS', '100')),
            rate_limit_window=int(os.getenv('RATE_LIMIT_WINDOW', '60')),
            enable_authentication=os.getenv('ENABLE_AUTHENTICATION', 'false').lower() == 'true',
            secret_key=os.getenv('SECRET_KEY'),
            allowed_origins=allowed_origins.split(',') if allowed_origins != '*' else ['*']
        )
        
        # Monitoring config
        self._configs['monitoring'] = MonitoringConfig(
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_format=os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            enable_metrics=os.getenv('ENABLE_METRICS', 'true').lower() == 'true',
            metrics_port=int(os.getenv('METRICS_PORT', '9090')),
            enable_tracing=os.getenv('ENABLE_TRACING', 'false').lower() == 'true',
            jaeger_endpoint=os.getenv('JAEGER_ENDPOINT')
        )
        
        # Server config
        cors_origins = os.getenv('CORS_ORIGINS', '*')
        self._configs['server'] = ServerConfig(
            host=os.getenv('HOST', '0.0.0.0'),
            port=int(os.getenv('PORT', '8001')),
            workers=int(os.getenv('WORKERS', '1')),
            reload=os.getenv('RELOAD', 'false').lower() == 'true',
            access_log=os.getenv('ACCESS_LOG', 'true').lower() == 'true',
            cors_origins=cors_origins.split(',') if cors_origins != '*' else ['*']
        )
    
    def _load_from_file(self):
        """Load configurations from config file."""
        try:
            with open(self.config_file, 'r') as f:
                if self.config_file.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    # Assume YAML
                    import yaml
                    config_data = yaml.safe_load(f)
            
            # Update configurations with file data
            for section, data in config_data.items():
                if section in self._configs:
                    config_obj = self._configs[section]
                    for key, value in data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
            
            logger.info(f"Configuration loaded from {self.config_file}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file {self.config_file}: {e}")
    
    def _validate_configs(self):
        """Validate all configurations."""
        # Validate AI config
        ai_config = self._configs['ai']
        if ai_config.provider in ['openai', 'anthropic', 'cohere'] and not ai_config.api_key:
            raise ConfigurationError(f"API key required for AI provider: {ai_config.provider}")
        
        # Validate processing config
        processing_config = self._configs['processing']
        if processing_config.max_file_size_mb <= 0:
            raise ConfigurationError("max_file_size_mb must be positive")
        
        if not processing_config.allowed_extensions:
            raise ConfigurationError("allowed_extensions cannot be empty")
        
        # Validate cache config
        cache_config = self._configs['cache']
        if cache_config.backend == 'redis' and not cache_config.redis_url:
            raise ConfigurationError("Redis URL required when using Redis cache backend")
        
        logger.info("All configurations validated successfully")
    
    def get(self, section: str) -> Any:
        """Get configuration section."""
        if section not in self._configs:
            raise ConfigurationError(f"Configuration section '{section}' not found")
        return self._configs[section]
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configurations."""
        return self._configs.copy()
    
    def update(self, section: str, **kwargs):
        """Update configuration section."""
        if section not in self._configs:
            raise ConfigurationError(f"Configuration section '{section}' not found")
        
        config_obj = self._configs[section]
        for key, value in kwargs.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
            else:
                raise ConfigurationError(f"Invalid configuration key: {key}")
        
        logger.info(f"Configuration section '{section}' updated")
    
    def save(self, file_path: Optional[Union[str, Path]] = None):
        """Save configurations to file."""
        save_path = Path(file_path) if file_path else self.config_file
        if not save_path:
            raise ConfigurationError("No file path provided for saving configuration")
        
        try:
            config_data = {}
            for section, config_obj in self._configs.items():
                config_data[section] = config_obj.__dict__
            
            with open(save_path, 'w') as f:
                if save_path.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2)
                else:
                    import yaml
                    yaml.dump(config_data, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {save_path}: {e}")


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(section: str) -> Any:
    """Get configuration section."""
    return get_config_manager().get(section)

















