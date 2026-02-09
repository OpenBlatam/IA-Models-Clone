from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
from typing import Optional, List, Dict, Any
from functools import lru_cache
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Optimized configuration for Key Messages feature.
"""


class RedisConfig(BaseSettings):
    """Redis configuration."""
    url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    max_connections: int = Field(default=20, description="Maximum Redis connections")
    decode_responses: bool = Field(default=True, description="Decode Redis responses")
    socket_keepalive: bool = Field(default=True, description="Enable socket keepalive")
    socket_keepalive_options: Dict[str, int] = Field(
        default={
            "TCP_KEEPIDLE": 1,
            "TCP_KEEPINTVL": 3,
            "TCP_KEEPCNT": 5
        },
        description="Socket keepalive options"
    )
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")

class HTTPConfig(BaseSettings):
    """HTTP client configuration."""
    timeout_seconds: int = Field(default=30, description="HTTP request timeout")
    max_keepalive_connections: int = Field(default=20, description="Max keepalive connections")
    max_connections: int = Field(default=100, description="Max total connections")
    http2: bool = Field(default=True, description="Enable HTTP/2")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_backoff_factor: float = Field(default=1.0, description="Retry backoff factor")
    
    model_config = SettingsConfigDict(env_prefix="HTTP_")

class CacheConfig(BaseSettings):
    """Cache configuration."""
    ttl_seconds: int = Field(default=86400, description="Cache TTL in seconds")
    memory_cache_size: int = Field(default=1000, description="Memory cache size")
    redis_enabled: bool = Field(default=True, description="Enable Redis caching")
    memory_enabled: bool = Field(default=True, description="Enable memory caching")
    
    model_config = SettingsConfigDict(env_prefix="CACHE_")

class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""
    enabled: bool = Field(default=True, description="Enable rate limiting")
    default_limit: str = Field(default="100/minute", description="Default rate limit")
    generate_limit: str = Field(default="50/minute", description="Generate endpoint limit")
    analyze_limit: str = Field(default="30/minute", description="Analyze endpoint limit")
    batch_limit: str = Field(default="10/minute", description="Batch endpoint limit")
    
    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_")

class CircuitBreakerConfig(BaseSettings):
    """Circuit breaker configuration."""
    enabled: bool = Field(default=True, description="Enable circuit breaker")
    failure_threshold: int = Field(default=5, description="Failure threshold")
    recovery_timeout: int = Field(default=60, description="Recovery timeout in seconds")
    expected_exception: List[str] = Field(
        default=["aiohttp.ClientError", "httpx.HTTPError"],
        description="Expected exceptions for circuit breaker"
    )
    
    model_config = SettingsConfigDict(env_prefix="CIRCUIT_BREAKER_")

class MonitoringConfig(BaseSettings):
    """Monitoring and metrics configuration."""
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    structured_logging: bool = Field(default=True, description="Enable structured logging")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    metrics_path: str = Field(default="/metrics", description="Prometheus metrics path")
    
    model_config = SettingsConfigDict(env_prefix="MONITORING_")

class LLMConfig(BaseSettings):
    """LLM service configuration."""
    provider: str = Field(default="openai", description="LLM provider")
    api_key: Optional[str] = Field(default=None, description="LLM API key")
    model: str = Field(default="gpt-3.5-turbo", description="LLM model")
    max_tokens: int = Field(default=1000, description="Maximum tokens")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    timeout_seconds: int = Field(default=30, description="LLM request timeout")
    
    model_config = SettingsConfigDict(env_prefix="LLM_")
    
    @validator('api_key', pre=True, always=True)
    async def validate_api_key(cls, v) -> bool:
        """Validate API key is provided."""
        if not v:
            # Try to get from environment
            v = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        return v

class SecurityConfig(BaseSettings):
    """Security configuration."""
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    cors_credentials: bool = Field(default=True, description="Allow CORS credentials")
    api_key_required: bool = Field(default=False, description="Require API key for requests")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    
    model_config = SettingsConfigDict(env_prefix="SECURITY_")

class PerformanceConfig(BaseSettings):
    """Performance optimization configuration."""
    max_batch_size: int = Field(default=50, description="Maximum batch size")
    concurrent_requests: int = Field(default=10, description="Concurrent request limit")
    compression_enabled: bool = Field(default=True, description="Enable response compression")
    compression_min_size: int = Field(default=1000, description="Minimum size for compression")
    
    model_config = SettingsConfigDict(env_prefix="PERFORMANCE_")

class KeyMessagesConfig(BaseSettings):
    """Main configuration for Key Messages feature."""
    
    # Environment
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Service configuration
    service_name: str = Field(default="key-messages", description="Service name")
    service_version: str = Field(default="2.0.0", description="Service version")
    
    # Sub-configurations
    redis: RedisConfig = Field(default_factory=RedisConfig)
    http: HTTPConfig = Field(default_factory=HTTPConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @validator('environment')
    def validate_environment(cls, v) -> bool:
        """Validate environment value."""
        valid_environments = ['development', 'staging', 'production', 'test']
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v
    
    @validator('debug')
    def validate_debug(cls, v, values) -> bool:
        """Validate debug mode based on environment."""
        if values.get('environment') == 'production' and v:
            raise ValueError("Debug mode cannot be enabled in production")
        return v
    
    def get_redis_url(self) -> str:
        """Get Redis URL with fallback."""
        if self.redis.url:
            return self.redis.url
        
        # Fallback to environment variables
        host = os.getenv('REDIS_HOST', 'localhost')
        port = os.getenv('REDIS_PORT', '6379')
        password = os.getenv('REDIS_PASSWORD')
        
        if password:
            return f"redis://:{password}@{host}:{port}"
        return f"redis://{host}:{port}"
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return {
            "provider": self.llm.provider,
            "api_key": self.llm.api_key,
            "model": self.llm.model,
            "max_tokens": self.llm.max_tokens,
            "temperature": self.llm.temperature,
            "timeout": self.llm.timeout_seconds
        }
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == 'development'
    
    def get_log_level(self) -> str:
        """Get appropriate log level."""
        if self.debug:
            return "DEBUG"
        elif self.is_production():
            return "INFO"
        else:
            return "DEBUG"

@lru_cache()
def get_settings() -> KeyMessagesConfig:
    """Get cached settings instance."""
    return KeyMessagesConfig()

# Convenience functions
def get_redis_config() -> RedisConfig:
    """Get Redis configuration."""
    return get_settings().redis

async def get_http_config() -> HTTPConfig:
    """Get HTTP configuration."""
    return get_settings().http

def get_cache_config() -> CacheConfig:
    """Get cache configuration."""
    return get_settings().cache

def get_rate_limit_config() -> RateLimitConfig:
    """Get rate limit configuration."""
    return get_settings().rate_limit

def get_circuit_breaker_config() -> CircuitBreakerConfig:
    """Get circuit breaker configuration."""
    return get_settings().circuit_breaker

def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration."""
    return get_settings().monitoring

def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    return get_settings().llm

def get_security_config() -> SecurityConfig:
    """Get security configuration."""
    return get_settings().security

def get_performance_config() -> PerformanceConfig:
    """Get performance configuration."""
    return get_settings().performance 

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration with guard clauses."""
    # Guard clauses for early validation
    if not config:
        raise ValueError("Configuration cannot be empty")
    
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
    
    # Validate required sections
    required_sections = ['app', 'models', 'training', 'data']
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")
    
    # Validate app configuration
    app_config = config.get('app', {})
    if not app_config.get('name'):
        raise ValueError("App name is required")
    
    if not app_config.get('version'):
        raise ValueError("App version is required")
    
    # Validate model configuration
    models_config = config.get('models', {})
    if not models_config:
        raise ValueError("Models configuration cannot be empty")
    
    for model_name, model_config in models_config.items():
        if not isinstance(model_config, dict):
            raise ValueError(f"Model configuration for {model_name} must be a dictionary")
        
        if not model_config.get('type'):
            raise ValueError(f"Model type is required for {model_name}")
        
        if model_config.get('max_length', 0) <= 0:
            raise ValueError(f"Max length must be positive for {model_name}")
    
    # Validate training configuration
    training_config = config.get('training', {})
    if training_config.get('batch_size', 0) <= 0:
        raise ValueError("Training batch size must be positive")
    
    if training_config.get('learning_rate', 0) <= 0:
        raise ValueError("Learning rate must be positive")
    
    if training_config.get('num_epochs', 0) <= 0:
        raise ValueError("Number of epochs must be positive")
    
    # Validate data configuration
    data_config = config.get('data', {})
    if data_config.get('max_file_size', 0) <= 0:
        raise ValueError("Max file size must be positive")
    
    # logger.info("Configuration validation passed") # This line was not in the original file, so it's not added.

def get_settings() -> Dict[str, Any]:
    """Get application settings with validation."""
    try:
        # Load configuration
        config = load_config()
        
        # Validate configuration
        validate_config(config)
        
        # Apply environment-specific overrides
        environment = os.getenv('ENVIRONMENT', 'development')
        config = apply_environment_overrides(config, environment)
        
        # logger.info("Settings loaded successfully", environment=environment) # This line was not in the original file, so it's not added.
        
        return config
        
    except Exception as e:
        # logger.error("Error loading settings", error=str(e)) # This line was not in the original file, so it's not added.
        raise

def apply_environment_overrides(config: Dict[str, Any], environment: str) -> Dict[str, Any]:
    """Apply environment-specific configuration overrides."""
    # Guard clauses for early validation
    if not environment or not environment.strip():
        raise ValueError("Environment cannot be empty")
    
    if environment not in ['development', 'production', 'testing']:
        raise ValueError(f"Invalid environment: {environment}")
    
    # Load environment-specific config
    env_config_path = f"config_{environment}.yaml"
    
    if os.path.exists(env_config_path):
        try:
            with open(env_config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                env_config = yaml.safe_load(f)
            
            # Merge configurations
            config = deep_merge(config, env_config)
            
            # logger.info("Environment overrides applied", environment=environment) # This line was not in the original file, so it's not added.
            
        except Exception as e:
            # logger.warning("Failed to load environment config", 
            #               environment=environment, error=str(e)) # This line was not in the original file, so it's not added.
            pass # Added pass to avoid NameError for logger
    
    return config

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries with validation."""
    # Guard clauses for early validation
    if not isinstance(base, dict):
        raise ValueError("Base configuration must be a dictionary")
    
    if not isinstance(override, dict):
        raise ValueError("Override configuration must be a dictionary")
    
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result 