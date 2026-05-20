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
from functools import lru_cache
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Application Settings
===================

Configuration management for the AI Video system.
"""




class DatabaseSettings(BaseModel):
    """Database configuration."""
    
    url: str = Field(default="postgresql://user:pass@localhost/ai_video")
    pool_size: int = Field(default=10, ge=1, le=50)
    max_overflow: int = Field(default=20, ge=0, le=100)
    echo: bool = Field(default=False)
    pool_pre_ping: bool = Field(default=True)
    pool_recycle: int = Field(default=3600)


class RedisSettings(BaseModel):
    """Redis configuration."""
    
    url: str = Field(default="redis://localhost:6379/0")
    max_connections: int = Field(default=20, ge=1, le=100)
    socket_timeout: float = Field(default=5.0, ge=1.0, le=30.0)
    socket_connect_timeout: float = Field(default=2.0, ge=1.0, le=10.0)
    retry_on_timeout: bool = Field(default=True)
    health_check_interval: int = Field(default=30, ge=10, le=300)


class CacheSettings(BaseModel):
    """Cache configuration."""
    
    redis: RedisSettings = Field(default_factory=RedisSettings)
    default_ttl: int = Field(default=3600, ge=60, le=86400)
    max_memory: str = Field(default="1gb")
    eviction_policy: str = Field(default="allkeys-lru")


class LoggingSettings(BaseModel):
    """Logging configuration."""
    
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    output: str = Field(default="stdout")
    file_path: Optional[str] = Field(default=None)
    max_size: str = Field(default="100MB")
    backup_count: int = Field(default=5, ge=1, le=10)
    enable_structured_logging: bool = Field(default=True)


class MetricsSettings(BaseModel):
    """Metrics configuration."""
    
    enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090, ge=1024, le=65535)
    push_gateway: Optional[str] = Field(default=None)
    interval: int = Field(default=15, ge=5, le=300)


class TracingSettings(BaseModel):
    """Tracing configuration."""
    
    enabled: bool = Field(default=True)
    service_name: str = Field(default="ai-video-system")
    endpoint: Optional[str] = Field(default=None)
    sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)


class SecuritySettings(BaseModel):
    """Security configuration."""
    
    secret_key: str = Field(default="your-secret-key-here")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30, ge=5, le=1440)
    refresh_token_expire_days: int = Field(default=7, ge=1, le=30)
    bcrypt_rounds: int = Field(default=12, ge=10, le=16)


class CORSettings(BaseModel):
    """CORS configuration."""
    
    origins: List[str] = Field(default=["http://localhost:3000"])
    allow_credentials: bool = Field(default=True)
    allow_methods: List[str] = Field(default=["*"])
    allow_headers: List[str] = Field(default=["*"])


class RateLimitSettings(BaseModel):
    """Rate limiting configuration."""
    
    enabled: bool = Field(default=True)
    default_limit: int = Field(default=100, ge=1, le=10000)
    window_size: int = Field(default=3600, ge=60, le=86400)
    storage_url: str = Field(default="redis://localhost:6379/1")


class MessagingSettings(BaseModel):
    """Messaging configuration."""
    
    redis: RedisSettings = Field(default_factory=RedisSettings)
    default_queue: str = Field(default="ai_video_queue")
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay: int = Field(default=60, ge=10, le=3600)


class ExternalServicesSettings(BaseModel):
    """External services configuration."""
    
    openai_api_key: Optional[str] = Field(default=None)
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    aws_access_key_id: Optional[str] = Field(default=None)
    aws_secret_access_key: Optional[str] = Field(default=None)
    aws_region: str = Field(default="us-east-1")
    s3_bucket: str = Field(default="ai-video-storage")


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)
    workers: int = Field(default=1, ge=1, le=16)
    
    # Database
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    
    # Cache
    cache: CacheSettings = Field(default_factory=CacheSettings)
    
    # Logging
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    # Metrics
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    
    # Tracing
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    
    # Security
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    
    # CORS
    cors: CORSettings = Field(default_factory=CORSettings)
    
    # Rate Limiting
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    
    # Messaging
    messaging: MessagingSettings = Field(default_factory=MessagingSettings)
    
    # External Services
    external_services: ExternalServicesSettings = Field(default_factory=ExternalServicesSettings)
    
    def __init__(self, **kwargs) -> Any:
        super().__init__(**kwargs)
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_from_env(self) -> Any:
        """Load settings from environment variables."""
        # Database
        if db_url := os.getenv("DATABASE_URL"):
            self.database.url = db_url
        
        # Redis
        if redis_url := os.getenv("REDIS_URL"):
            self.cache.redis.url = redis_url
            self.messaging.redis.url = redis_url
        
        # Security
        if secret_key := os.getenv("SECRET_KEY"):
            self.security.secret_key = secret_key
        
        # External services
        if openai_key := os.getenv("OPENAI_API_KEY"):
            self.external_services.openai_api_key = openai_key
        
        if aws_key := os.getenv("AWS_ACCESS_KEY_ID"):
            self.external_services.aws_access_key_id = aws_key
        
        if aws_secret := os.getenv("AWS_SECRET_ACCESS_KEY"):
            self.external_services.aws_secret_access_key = aws_secret
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == "testing"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_database_url() -> str:
    """Get database URL from settings."""
    return get_settings().database.url


def get_redis_url() -> str:
    """Get Redis URL from settings."""
    return get_settings().cache.redis.url


def get_secret_key() -> str:
    """Get secret key from settings."""
    return get_settings().security.secret_key 