from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
from typing import Optional, List
from pydantic import BaseSettings, Field
from functools import lru_cache
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Settings Configuration
=====================

Centralized configuration management for the copywriting system.
"""



class EngineConfig(BaseSettings):
    """Engine configuration settings"""
    
    # Performance settings
    max_workers: int = Field(default=8, description="Maximum worker threads")
    max_batch_size: int = Field(default=64, description="Maximum batch size")
    cache_ttl: int = Field(default=7200, description="Cache TTL in seconds")
    max_cache_size: int = Field(default=50000, description="Maximum cache entries")
    
    # GPU and optimization settings
    enable_gpu: bool = Field(default=True, description="Enable GPU acceleration")
    enable_quantization: bool = Field(default=True, description="Enable model quantization")
    enable_mixed_precision: bool = Field(default=True, description="Enable FP16 operations")
    enable_batching: bool = Field(default=True, description="Enable batch processing")
    enable_caching: bool = Field(default=True, description="Enable caching")
    
    # Memory optimization
    enable_memory_optimization: bool = Field(default=True, description="Enable memory optimization")
    max_memory_usage: float = Field(default=0.8, description="Maximum memory usage (80%)")
    gc_threshold: int = Field(default=1000, description="Garbage collection threshold")
    
    # Model settings
    default_model: str = Field(default="gpt2-medium", description="Default model")
    fallback_model: str = Field(default="distilgpt2", description="Fallback model")
    max_tokens: int = Field(default=1024, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Generation temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling")
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty")
    
    # Cache settings
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    cache_prefix: str = Field(default="copywriting:", description="Cache key prefix")
    enable_compression: bool = Field(default=True, description="Enable cache compression")
    
    # Timeout settings
    request_timeout: float = Field(default=45.0, description="Request timeout in seconds")
    batch_timeout: float = Field(default=120.0, description="Batch timeout in seconds")
    
    @dataclass
class Config:
        env_prefix = "ENGINE_"


class APIConfig(BaseSettings):
    """API configuration settings"""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")
    reload: bool = Field(default=False, description="Enable auto-reload")
    
    # API settings
    enable_docs: bool = Field(default=True, description="Enable API documentation")
    enable_cors: bool = Field(default=True, description="Enable CORS")
    enable_gzip: bool = Field(default=True, description="Enable GZIP compression")
    
    # Rate limiting
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    max_requests_per_minute: int = Field(default=100, description="Max requests per minute")
    
    @dataclass
class Config:
        env_prefix = "API_"


class SecurityConfig(BaseSettings):
    """Security configuration settings"""
    
    # Authentication
    secret_key: str = Field(default="your-secret-key-change-this", description="Secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Token expiration")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], description="CORS origins")
    cors_credentials: bool = Field(default=True, description="CORS credentials")
    cors_methods: List[str] = Field(default=["*"], description="CORS methods")
    cors_headers: List[str] = Field(default=["*"], description="CORS headers")
    
    # Input validation
    enable_input_validation: bool = Field(default=True, description="Enable input validation")
    max_request_size: int = Field(default=10485760, description="Max request size (10MB)")
    
    @dataclass
class Config:
        env_prefix = "SECURITY_"


class MonitoringConfig(BaseSettings):
    """Monitoring configuration settings"""
    
    # Metrics
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_profiling: bool = Field(default=True, description="Enable performance profiling")
    enable_memory_monitoring: bool = Field(default=True, description="Enable memory monitoring")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format")
    enable_structured_logging: bool = Field(default=True, description="Enable structured logging")
    
    # Health checks
    health_check_interval: int = Field(default=30, description="Health check interval")
    enable_detailed_health: bool = Field(default=True, description="Enable detailed health checks")
    
    @dataclass
class Config:
        env_prefix = "MONITORING_"


class Settings(BaseSettings):
    """Main application settings"""
    
    # Environment
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Configurations
    engine: EngineConfig = Field(default_factory=EngineConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    @dataclass
class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def get_engine_config() -> EngineConfig:
    """Get engine configuration"""
    return get_settings().engine


async def get_api_config() -> APIConfig:
    """Get API configuration"""
    return get_settings().api


def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return get_settings().security


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return get_settings().monitoring 