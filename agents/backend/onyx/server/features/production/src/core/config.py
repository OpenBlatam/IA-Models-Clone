from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
TIMEOUT_SECONDS = 60

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from pydantic.types import SecretStr
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🔧 Configuration Management Module
=================================

Production-grade configuration with environment variables,
validation, and type safety using Pydantic.
"""



class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    
    URL: str = Field(default="postgresql://user:pass@localhost/db", env="DATABASE_URL")
    POOL_SIZE: int = Field(default=20, env="DB_POOL_SIZE")
    MAX_OVERFLOW: int = Field(default=30, env="DB_MAX_OVERFLOW")
    POOL_TIMEOUT: int = Field(default=30, env="DB_POOL_TIMEOUT")
    POOL_RECYCLE: int = Field(default=3600, env="DB_POOL_RECYCLE")
    ECHO: bool = Field(default=False, env="DB_ECHO")
    
    @dataclass
class Config:
        env_prefix = "DB_"


class RedisSettings(BaseSettings):
    """Redis configuration settings"""
    
    URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    PASSWORD: Optional[SecretStr] = Field(default=None, env="REDIS_PASSWORD")
    DB: int = Field(default=0, env="REDIS_DB")
    MAX_CONNECTIONS: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    SOCKET_TIMEOUT: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    SOCKET_CONNECT_TIMEOUT: int = Field(default=5, env="REDIS_SOCKET_CONNECT_TIMEOUT")
    
    @dataclass
class Config:
        env_prefix = "REDIS_"


class OpenAISettings(BaseSettings):
    """OpenAI configuration settings"""
    
    API_KEY: SecretStr = Field(..., env="OPENAI_API_KEY")
    ORGANIZATION: Optional[str] = Field(default=None, env="OPENAI_ORGANIZATION")
    BASE_URL: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")
    TIMEOUT: int = Field(default=60, env="OPENAI_TIMEOUT")
    MAX_RETRIES: int = Field(default=3, env="OPENAI_MAX_RETRIES")
    MODEL: str = Field(default="gpt-4", env="OPENAI_MODEL")
    MAX_TOKENS: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    TEMPERATURE: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    
    @dataclass
class Config:
        env_prefix = "OPENAI_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings"""
    
    ENABLED: bool = Field(default=True, env="MONITORING_ENABLED")
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    JAEGER_ENDPOINT: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    JAEGER_HOST: str = Field(default="localhost", env="JAEGER_HOST")
    JAEGER_PORT: int = Field(default=6831, env="JAEGER_PORT")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    STRUCTURED_LOGGING: bool = Field(default=True, env="STRUCTURED_LOGGING")
    
    @dataclass
class Config:
        env_prefix = "MONITORING_"


class SecuritySettings(BaseSettings):
    """Security configuration settings"""
    
    SECRET_KEY: SecretStr = Field(..., env="SECRET_KEY")
    ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    RATE_LIMIT_PER_MINUTE: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    CORS_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    ALLOW_CREDENTIALS: bool = Field(default=True, env="ALLOW_CREDENTIALS")
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v) -> Any:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @dataclass
class Config:
        env_prefix = "SECURITY_"


class CacheSettings(BaseSettings):
    """Cache configuration settings"""
    
    ENABLED: bool = Field(default=True, env="CACHE_ENABLED")
    TTL: int = Field(default=3600, env="CACHE_TTL")
    MAX_SIZE: int = Field(default=1000, env="CACHE_MAX_SIZE")
    COMPRESSION: bool = Field(default=True, env="CACHE_COMPRESSION")
    PERSISTENCE: bool = Field(default=False, env="CACHE_PERSISTENCE")
    
    @dataclass
class Config:
        env_prefix = "CACHE_"


class AISettings(BaseSettings):
    """AI service configuration settings"""
    
    ENABLED: bool = Field(default=True, env="AI_ENABLED")
    GPU_ENABLED: bool = Field(default=False, env="AI_GPU_ENABLED")
    BATCH_SIZE: int = Field(default=10, env="AI_BATCH_SIZE")
    MAX_CONCURRENT_REQUESTS: int = Field(default=50, env="AI_MAX_CONCURRENT_REQUESTS")
    TIMEOUT: int = Field(default=120, env="AI_TIMEOUT")
    RETRY_ATTEMPTS: int = Field(default=3, env="AI_RETRY_ATTEMPTS")
    FALLBACK_ENABLED: bool = Field(default=True, env="AI_FALLBACK_ENABLED")
    
    @dataclass
class Config:
        env_prefix = "AI_"


class EventSettings(BaseSettings):
    """Event system configuration settings"""
    
    ENABLED: bool = Field(default=True, env="EVENTS_ENABLED")
    BROKER_URL: str = Field(default="redis://localhost:6379/1", env="EVENT_BROKER_URL")
    QUEUE_NAME: str = Field(default="ai_events", env="EVENT_QUEUE_NAME")
    MAX_RETRIES: int = Field(default=3, env="EVENT_MAX_RETRIES")
    BATCH_SIZE: int = Field(default=100, env="EVENT_BATCH_SIZE")
    
    @dataclass
class Config:
        env_prefix = "EVENT_"


class Settings(BaseSettings):
    """Main application settings"""
    
    # Application
    APP_NAME: str = Field(default="Ultra-Optimized AI Copywriting System", env="APP_NAME")
    VERSION: str = Field(default="2.0.0", env="VERSION")
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")
    RELOAD: bool = Field(default=False, env="RELOAD")
    
    # Performance
    MAX_CONNECTIONS: int = Field(default=1000, env="MAX_CONNECTIONS")
    KEEP_ALIVE: int = Field(default=5, env="KEEP_ALIVE")
    BACKLOG: int = Field(default=2048, env="BACKLOG")
    
    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    openai: OpenAISettings = OpenAISettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    security: SecuritySettings = SecuritySettings()
    cache: CacheSettings = CacheSettings()
    ai: AISettings = AISettings()
    events: EventSettings = EventSettings()
    
    # Computed properties
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() in ["development", "dev", "local"]
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() in ["production", "prod"]
    
    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT.lower() in ["testing", "test"]
    
    @property
    def log_level(self) -> str:
        return self.monitoring.LOG_LEVEL
    
    @property
    def cors_origins(self) -> List[str]:
        return self.security.CORS_ORIGINS
    
    @dataclass
class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        validate_assignment = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment"""
    global settings
    settings = Settings()
    return settings 