from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from pydantic import BaseSettings, Field, validator
from typing import Optional, List, Dict, Any
import os
from functools import lru_cache
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Advanced Configuration for LinkedIn Posts API
============================================

Environment-based configuration with validation and defaults.
"""



class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    """
    
    # API Configuration
    API_VERSION: str = Field("2.0.0", env="API_VERSION")
    API_TITLE: str = Field("LinkedIn Posts API", env="API_TITLE")
    API_DESCRIPTION: str = Field(
        "High-performance LinkedIn post management API with NLP enhancement",
        env="API_DESCRIPTION"
    )
    API_PREFIX: str = Field("/api/v2", env="API_PREFIX")
    
    # Server Configuration
    HOST: str = Field("0.0.0.0", env="HOST")
    PORT: int = Field(8000, env="PORT")
    WORKERS: int = Field(4, env="WORKERS")
    RELOAD: bool = Field(False, env="RELOAD")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    
    # Performance Configuration
    MAX_CONNECTIONS: int = Field(1000, env="MAX_CONNECTIONS")
    CONNECTION_TIMEOUT: int = Field(30, env="CONNECTION_TIMEOUT")
    KEEPALIVE_TIMEOUT: int = Field(65, env="KEEPALIVE_TIMEOUT")
    REQUEST_TIMEOUT: int = Field(60, env="REQUEST_TIMEOUT")
    
    # Cache Configuration
    REDIS_URL: str = Field(
        "redis://localhost:6379/0",
        env="REDIS_URL"
    )
    CACHE_TTL: int = Field(300, env="CACHE_TTL")
    CACHE_MAX_SIZE: int = Field(10000, env="CACHE_MAX_SIZE")
    ENABLE_CACHE: bool = Field(True, env="ENABLE_CACHE")
    
    # Database Configuration
    DATABASE_URL: str = Field(
        "postgresql://user:pass@localhost/linkedin_posts",
        env="DATABASE_URL"
    )
    DATABASE_POOL_SIZE: int = Field(20, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(40, env="DATABASE_MAX_OVERFLOW")
    DATABASE_POOL_TIMEOUT: int = Field(30, env="DATABASE_POOL_TIMEOUT")
    
    # Security Configuration
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = Field("HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        ["http://localhost:3000", "https://app.example.com"],
        env="CORS_ORIGINS"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(True, env="CORS_ALLOW_CREDENTIALS")
    CORS_ALLOW_METHODS: List[str] = Field(["*"], env="CORS_ALLOW_METHODS")
    CORS_ALLOW_HEADERS: List[str] = Field(["*"], env="CORS_ALLOW_HEADERS")
    
    # Rate Limiting Configuration
    RATE_LIMIT_ENABLED: bool = Field(True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_CALLS: int = Field(100, env="RATE_LIMIT_CALLS")
    RATE_LIMIT_PERIOD: int = Field(60, env="RATE_LIMIT_PERIOD")
    RATE_LIMIT_BURST: int = Field(20, env="RATE_LIMIT_BURST")
    
    # NLP Configuration
    NLP_MODEL_NAME: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        env="NLP_MODEL_NAME"
    )
    NLP_BATCH_SIZE: int = Field(32, env="NLP_BATCH_SIZE")
    NLP_MAX_LENGTH: int = Field(512, env="NLP_MAX_LENGTH")
    NLP_CACHE_SIZE: int = Field(1000, env="NLP_CACHE_SIZE")
    ENABLE_FAST_NLP: bool = Field(True, env="ENABLE_FAST_NLP")
    ENABLE_ASYNC_NLP: bool = Field(True, env="ENABLE_ASYNC_NLP")
    
    # LangChain Configuration
    LANGCHAIN_API_KEY: Optional[str] = Field(None, env="LANGCHAIN_API_KEY")
    LANGCHAIN_MODEL: str = Field("gpt-3.5-turbo", env="LANGCHAIN_MODEL")
    LANGCHAIN_TEMPERATURE: float = Field(0.7, env="LANGCHAIN_TEMPERATURE")
    LANGCHAIN_MAX_TOKENS: int = Field(500, env="LANGCHAIN_MAX_TOKENS")
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    OPENAI_ORG_ID: Optional[str] = Field(None, env="OPENAI_ORG_ID")
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = Field(True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(9090, env="METRICS_PORT")
    ENABLE_TRACING: bool = Field(True, env="ENABLE_TRACING")
    JAEGER_ENDPOINT: str = Field(
        "http://localhost:14268/api/traces",
        env="JAEGER_ENDPOINT"
    )
    
    # Feature Flags
    ENABLE_BATCH_OPERATIONS: bool = Field(True, env="ENABLE_BATCH_OPERATIONS")
    ENABLE_STREAMING: bool = Field(True, env="ENABLE_STREAMING")
    ENABLE_WEBSOCKETS: bool = Field(True, env="ENABLE_WEBSOCKETS")
    ENABLE_COMPRESSION: bool = Field(True, env="ENABLE_COMPRESSION")
    
    # Optimization Configuration
    USE_UVLOOP: bool = Field(True, env="USE_UVLOOP")
    ENABLE_RESPONSE_COMPRESSION: bool = Field(True, env="ENABLE_RESPONSE_COMPRESSION")
    COMPRESSION_LEVEL: int = Field(6, env="COMPRESSION_LEVEL")
    MIN_COMPRESSION_SIZE: int = Field(1024, env="MIN_COMPRESSION_SIZE")
    
    # Batch Processing Configuration
    BATCH_MAX_SIZE: int = Field(100, env="BATCH_MAX_SIZE")
    BATCH_TIMEOUT: int = Field(30, env="BATCH_TIMEOUT")
    BATCH_CONCURRENT_LIMIT: int = Field(10, env="BATCH_CONCURRENT_LIMIT")
    
    # Health Check Configuration
    HEALTH_CHECK_INTERVAL: int = Field(30, env="HEALTH_CHECK_INTERVAL")
    HEALTH_CHECK_TIMEOUT: int = Field(5, env="HEALTH_CHECK_TIMEOUT")
    
    # Development Configuration
    DEBUG: bool = Field(False, env="DEBUG")
    TESTING: bool = Field(False, env="TESTING")
    ENVIRONMENT: str = Field("production", env="ENVIRONMENT")
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v) -> Any:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v) -> bool:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v) -> bool:
        valid_envs = ["development", "staging", "production", "testing"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment. Must be one of: {valid_envs}")
        return v.lower()
    
    @dataclass
class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def get_redis_settings(self) -> Dict[str, Any]:
        """Get Redis connection settings."""
        return {
            "url": self.REDIS_URL,
            "encoding": "utf-8",
            "decode_responses": False,
            "max_connections": 50,
            "socket_keepalive": True,
            "socket_keepalive_options": {
                1: 1,  # TCP_KEEPIDLE
                2: 1,  # TCP_KEEPINTVL
                3: 3,  # TCP_KEEPCNT
            }
        }
    
    def get_database_settings(self) -> Dict[str, Any]:
        """Get database connection settings."""
        return {
            "url": self.DATABASE_URL,
            "pool_size": self.DATABASE_POOL_SIZE,
            "max_overflow": self.DATABASE_MAX_OVERFLOW,
            "pool_timeout": self.DATABASE_POOL_TIMEOUT,
            "pool_pre_ping": True,
            "echo": self.DEBUG,
        }
    
    def get_cors_settings(self) -> Dict[str, Any]:
        """Get CORS middleware settings."""
        return {
            "allow_origins": self.CORS_ORIGINS,
            "allow_credentials": self.CORS_ALLOW_CREDENTIALS,
            "allow_methods": self.CORS_ALLOW_METHODS,
            "allow_headers": self.CORS_ALLOW_HEADERS,
        }
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENVIRONMENT == "production"
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.ENVIRONMENT == "development"
    
    def is_testing(self) -> bool:
        """Check if running in testing."""
        return self.ENVIRONMENT == "testing" or self.TESTING


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    """
    return Settings()


# Create settings instance
settings = get_settings()


# Configuration presets for different environments
ENVIRONMENT_CONFIGS = {
    "development": {
        "DEBUG": True,
        "RELOAD": True,
        "LOG_LEVEL": "DEBUG",
        "WORKERS": 1,
        "ENABLE_METRICS": False,
        "ENABLE_TRACING": False,
    },
    "staging": {
        "DEBUG": False,
        "RELOAD": False,
        "LOG_LEVEL": "INFO",
        "WORKERS": 2,
        "ENABLE_METRICS": True,
        "ENABLE_TRACING": True,
    },
    "production": {
        "DEBUG": False,
        "RELOAD": False,
        "LOG_LEVEL": "WARNING",
        "WORKERS": 4,
        "ENABLE_METRICS": True,
        "ENABLE_TRACING": True,
        "USE_UVLOOP": True,
        "ENABLE_COMPRESSION": True,
    },
    "testing": {
        "DEBUG": True,
        "TESTING": True,
        "LOG_LEVEL": "DEBUG",
        "WORKERS": 1,
        "ENABLE_CACHE": False,
        "RATE_LIMIT_ENABLED": False,
    }
}


def load_environment_config(environment: str) -> None:
    """
    Load configuration preset for environment.
    """
    if environment in ENVIRONMENT_CONFIGS:
        config = ENVIRONMENT_CONFIGS[environment]
        for key, value in config.items():
            os.environ[key] = str(value)


# Export
__all__ = [
    "Settings",
    "settings",
    "get_settings",
    "load_environment_config",
    "ENVIRONMENT_CONFIGS"
] 