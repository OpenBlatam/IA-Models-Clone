from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, Any
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Redis Configuration - Onyx Integration
Configuration settings for Redis integration in Onyx.
"""

class RedisConfig(BaseModel):
    """Redis configuration settings."""
    
    # Connection settings
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: str = Field(default="", description="Redis password")
    
    # Connection pool settings
    max_connections: int = Field(default=10, description="Maximum number of connections in the pool")
    socket_timeout: int = Field(default=5, description="Socket timeout in seconds")
    socket_connect_timeout: int = Field(default=5, description="Socket connection timeout in seconds")
    
    # Cache settings
    default_expire: int = Field(default=3600, description="Default cache expiration time in seconds")
    model_expire: int = Field(default=86400, description="Model cache expiration time in seconds")
    context_expire: int = Field(default=1800, description="Context cache expiration time in seconds")
    prompt_expire: int = Field(default=86400, description="Prompt cache expiration time in seconds")
    metrics_expire: int = Field(default=3600, description="Metrics cache expiration time in seconds")
    
    # Prefix settings
    model_prefix: str = Field(default="model", description="Prefix for model cache keys")
    context_prefix: str = Field(default="context", description="Prefix for context cache keys")
    prompt_prefix: str = Field(default="prompt", description="Prefix for prompt cache keys")
    metrics_prefix: str = Field(default="metrics", description="Prefix for metrics cache keys")
    counter_prefix: str = Field(default="counter", description="Prefix for counter keys")
    set_prefix: str = Field(default="set", description="Prefix for set keys")
    
    # Retry settings
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    retry_delay: int = Field(default=1, description="Delay between retries in seconds")
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                          description="Logging format")
    
    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True

# Default configuration
default_config = RedisConfig()

# Environment-specific configurations
configurations: Dict[str, RedisConfig] = {
    "development": RedisConfig(
        host="localhost",
        port=6379,
        db=0,
        default_expire=3600,
        log_level="DEBUG"
    ),
    "testing": RedisConfig(
        host="localhost",
        port=6379,
        db=15,
        default_expire=300,
        log_level="DEBUG"
    ),
    "production": RedisConfig(
        host="redis.production",
        port=6379,
        db=0,
        password="your-secure-password",
        max_connections=50,
        default_expire=86400,
        log_level="INFO"
    )
}

def get_config(environment: str = "development") -> RedisConfig:
    """Get Redis configuration for the specified environment."""
    return configurations.get(environment, default_config)

# Example usage:
"""
# Get configuration for current environment
config = get_config()

# Use configuration in Redis manager
redis_manager = RedisManager(
    host=config.host,
    port=config.port,
    db=config.db,
    password=config.password
)

# Use configuration for caching
redis_manager.cache_model(
    model=brand_voice,
    prefix=config.model_prefix,
    identifier='brand_123',
    expire=config.model_expire
)

# Use configuration for context caching
redis_manager.cache_context(
    context={'user_id': '123'},
    prefix=config.context_prefix,
    identifier='user_123',
    expire=config.context_expire
)

# Use configuration for prompt caching
redis_manager.cache_prompt(
    prompt='Generate content',
    prefix=config.prompt_prefix,
    identifier='content_123',
    expire=config.prompt_expire
)

# Use configuration for metrics caching
redis_manager.cache_metrics(
    metrics={'clicks': 100},
    prefix=config.metrics_prefix,
    identifier='campaign_123',
    expire=config.metrics_expire
)
""" 