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
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging
from typing import Any, List, Dict, Optional
import asyncio
"""
Optimized Configuration System
=============================

Advanced configuration management with:
- Environment-based settings
- Performance optimization
- Security features
- Monitoring configuration
"""


logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "postgresql://localhost/copywriting"
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class RedisConfig:
    """Redis configuration"""
    url: str = "redis://localhost:6379"
    pool_size: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True


@dataclass
class ModelConfig:
    """ML model configuration"""
    default_model: str = "gpt2"
    fallback_model: str = "distilgpt2"
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    enable_gpu: bool = True
    enable_quantization: bool = True
    model_cache_dir: str = "./models"


@dataclass
class PerformanceConfig:
    """Performance configuration"""
    max_workers: int = 4
    max_batch_size: int = 32
    request_timeout: float = 30.0
    batch_timeout: float = 60.0
    enable_caching: bool = True
    cache_ttl: int = 3600
    max_cache_size: int = 10000
    enable_compression: bool = True
    compression_level: int = 6


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600
    enable_cors: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_metrics: bool = True
    enable_profiling: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    metrics_port: int = 9090
    health_check_interval: int = 30
    enable_alerting: bool = False
    alert_webhook_url: Optional[str] = None


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    enable_docs: bool = True
    enable_reload: bool = False
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    enable_gzip: bool = True
    enable_cors: bool = True


class OptimizedConfig:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        
    """__init__ function."""
self.config_path = config_path or "config.json"
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Initialize configurations
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.model = ModelConfig()
        self.performance = PerformanceConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.api = APIConfig()
        
        # Load configuration
        self._load_config()
        self._load_environment_vars()
        
        logger.info(f"Configuration loaded for environment: {self.environment}")
    
    def _load_config(self) -> Any:
        """Load configuration from file"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    config_data = json.load(f)
                
                # Update configurations based on environment
                env_config = config_data.get(self.environment, {})
                
                # Update database config
                if 'database' in env_config:
                    db_config = env_config['database']
                    self.database = DatabaseConfig(**db_config)
                
                # Update redis config
                if 'redis' in env_config:
                    redis_config = env_config['redis']
                    self.redis = RedisConfig(**redis_config)
                
                # Update model config
                if 'model' in env_config:
                    model_config = env_config['model']
                    self.model = ModelConfig(**model_config)
                
                # Update performance config
                if 'performance' in env_config:
                    perf_config = env_config['performance']
                    self.performance = PerformanceConfig(**perf_config)
                
                # Update security config
                if 'security' in env_config:
                    sec_config = env_config['security']
                    self.security = SecurityConfig(**sec_config)
                
                # Update monitoring config
                if 'monitoring' in env_config:
                    mon_config = env_config['monitoring']
                    self.monitoring = MonitoringConfig(**mon_config)
                
                # Update API config
                if 'api' in env_config:
                    api_config = env_config['api']
                    self.api = APIConfig(**api_config)
                
                logger.info("Configuration loaded from file")
            
        except Exception as e:
            logger.warning(f"Error loading config file: {e}")
    
    def _load_environment_vars(self) -> Any:
        """Load configuration from environment variables"""
        # Database
        if os.getenv("DATABASE_URL"):
            self.database.url = os.getenv("DATABASE_URL")
        
        # Redis
        if os.getenv("REDIS_URL"):
            self.redis.url = os.getenv("REDIS_URL")
        
        # Model
        if os.getenv("DEFAULT_MODEL"):
            self.model.default_model = os.getenv("DEFAULT_MODEL")
        
        if os.getenv("ENABLE_GPU"):
            self.model.enable_gpu = os.getenv("ENABLE_GPU").lower() == "true"
        
        # Performance
        if os.getenv("MAX_WORKERS"):
            self.performance.max_workers = int(os.getenv("MAX_WORKERS"))
        
        if os.getenv("CACHE_TTL"):
            self.performance.cache_ttl = int(os.getenv("CACHE_TTL"))
        
        # Security
        if os.getenv("SECRET_KEY"):
            self.security.secret_key = os.getenv("SECRET_KEY")
        
        # API
        if os.getenv("API_HOST"):
            self.api.host = os.getenv("API_HOST")
        
        if os.getenv("API_PORT"):
            self.api.port = int(os.getenv("API_PORT"))
        
        # Monitoring
        if os.getenv("LOG_LEVEL"):
            self.monitoring.log_level = os.getenv("LOG_LEVEL")
        
        logger.info("Environment variables loaded")
    
    def get_database_url(self) -> str:
        """Get database URL"""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Get Redis URL"""
        return self.redis.url
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
    def is_testing(self) -> bool:
        """Check if running in testing"""
        return self.environment == "testing"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': self.environment,
            'database': {
                'url': self.database.url,
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow,
                'pool_timeout': self.database.pool_timeout,
                'pool_recycle': self.database.pool_recycle
            },
            'redis': {
                'url': self.redis.url,
                'pool_size': self.redis.pool_size,
                'socket_timeout': self.redis.socket_timeout,
                'socket_connect_timeout': self.redis.socket_connect_timeout,
                'retry_on_timeout': self.redis.retry_on_timeout
            },
            'model': {
                'default_model': self.model.default_model,
                'fallback_model': self.model.fallback_model,
                'max_tokens': self.model.max_tokens,
                'temperature': self.model.temperature,
                'top_p': self.model.top_p,
                'enable_gpu': self.model.enable_gpu,
                'enable_quantization': self.model.enable_quantization,
                'model_cache_dir': self.model.model_cache_dir
            },
            'performance': {
                'max_workers': self.performance.max_workers,
                'max_batch_size': self.performance.max_batch_size,
                'request_timeout': self.performance.request_timeout,
                'batch_timeout': self.performance.batch_timeout,
                'enable_caching': self.performance.enable_caching,
                'cache_ttl': self.performance.cache_ttl,
                'max_cache_size': self.performance.max_cache_size,
                'enable_compression': self.performance.enable_compression,
                'compression_level': self.performance.compression_level
            },
            'security': {
                'algorithm': self.security.algorithm,
                'access_token_expire_minutes': self.security.access_token_expire_minutes,
                'enable_rate_limiting': self.security.enable_rate_limiting,
                'rate_limit_requests': self.security.rate_limit_requests,
                'rate_limit_window': self.security.rate_limit_window,
                'enable_cors': self.security.enable_cors,
                'cors_origins': self.security.cors_origins
            },
            'monitoring': {
                'enable_metrics': self.monitoring.enable_metrics,
                'enable_profiling': self.monitoring.enable_profiling,
                'enable_logging': self.monitoring.enable_logging,
                'log_level': self.monitoring.log_level,
                'metrics_port': self.monitoring.metrics_port,
                'health_check_interval': self.monitoring.health_check_interval,
                'enable_alerting': self.monitoring.enable_alerting,
                'alert_webhook_url': self.monitoring.alert_webhook_url
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'workers': self.api.workers,
                'enable_docs': self.api.enable_docs,
                'enable_reload': self.api.enable_reload,
                'max_request_size': self.api.max_request_size,
                'enable_gzip': self.api.enable_gzip,
                'enable_cors': self.api.enable_cors
            }
        }
    
    def save_config(self, config_path: Optional[str] = None):
        """Save configuration to file"""
        config_path = config_path or self.config_path
        
        try:
            config_data = {
                self.environment: self.to_dict()
            }
            
            with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def validate(self) -> bool:
        """Validate configuration"""
        try:
            # Validate required fields
            if not self.database.url:
                logger.error("Database URL is required")
                return False
            
            if not self.redis.url:
                logger.error("Redis URL is required")
                return False
            
            if not self.model.default_model:
                logger.error("Default model is required")
                return False
            
            if self.performance.max_workers <= 0:
                logger.error("Max workers must be greater than 0")
                return False
            
            if self.api.port <= 0 or self.api.port > 65535:
                logger.error("API port must be between 1 and 65535")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


# Global configuration instance
config = OptimizedConfig()

# Convenience functions
def get_config() -> OptimizedConfig:
    """Get global configuration instance"""
    return config

def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return config.database

def get_redis_config() -> RedisConfig:
    """Get Redis configuration"""
    return config.redis

def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return config.model

def get_performance_config() -> PerformanceConfig:
    """Get performance configuration"""
    return config.performance

def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return config.security

def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return config.monitoring

async def get_api_config() -> APIConfig:
    """Get API configuration"""
    return config.api 