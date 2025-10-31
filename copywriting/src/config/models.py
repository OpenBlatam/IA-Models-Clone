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

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuration Models
===================

Pydantic models for configuration validation and management.
"""



class EngineConfig(BaseModel):
    """Engine configuration model"""
    
    # Performance settings
    max_workers: int = Field(default=8, ge=1, le=32)
    max_batch_size: int = Field(default=64, ge=1, le=128)
    cache_ttl: int = Field(default=7200, ge=60, le=86400)
    max_cache_size: int = Field(default=50000, ge=1000, le=1000000)
    
    # GPU and optimization settings
    enable_gpu: bool = True
    enable_quantization: bool = True
    enable_mixed_precision: bool = True
    enable_batching: bool = True
    enable_caching: bool = True
    
    # Memory optimization
    enable_memory_optimization: bool = True
    max_memory_usage: float = Field(default=0.8, ge=0.1, le=0.95)
    gc_threshold: int = Field(default=1000, ge=100, le=10000)
    
    # Model settings
    default_model: str = "gpt2-medium"
    fallback_model: str = "distilgpt2"
    max_tokens: int = Field(default=1024, ge=64, le=4096)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    
    # Cache settings
    redis_url: str = "redis://localhost:6379"
    cache_prefix: str = "copywriting:"
    enable_compression: bool = True
    
    # Timeout settings
    request_timeout: float = Field(default=45.0, ge=5.0, le=300.0)
    batch_timeout: float = Field(default=120.0, ge=10.0, le=600.0)


class APIConfig(BaseModel):
    """API configuration model"""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1024, le=65535)
    workers: int = Field(default=1, ge=1, le=16)
    reload: bool = False
    
    # API settings
    enable_docs: bool = True
    enable_cors: bool = True
    enable_gzip: bool = True
    
    # Rate limiting
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = Field(default=100, ge=10, le=10000)


class SecurityConfig(BaseModel):
    """Security configuration model"""
    
    # Authentication
    secret_key: str = "your-secret-key-change-this"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=30, ge=5, le=1440)
    
    # CORS settings
    cors_origins: List[str] = ["*"]
    cors_credentials: bool = True
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    # Input validation
    enable_input_validation: bool = True
    max_request_size: int = Field(default=10485760, ge=1024, le=104857600)


class MonitoringConfig(BaseModel):
    """Monitoring configuration model"""
    
    # Metrics
    enable_metrics: bool = True
    enable_profiling: bool = True
    enable_memory_monitoring: bool = True
    
    # Logging
    log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_format: str = Field(default="json", regex="^(json|text)$")
    enable_structured_logging: bool = True
    
    # Health checks
    health_check_interval: int = Field(default=30, ge=5, le=300)
    enable_detailed_health: bool = True


class DatabaseConfig(BaseModel):
    """Database configuration model"""
    
    # Database settings
    database_url: str = "sqlite:///./copywriting.db"
    echo: bool = False
    pool_size: int = Field(default=10, ge=1, le=50)
    max_overflow: int = Field(default=20, ge=0, le=100)
    pool_timeout: int = Field(default=30, ge=5, le=300)
    pool_recycle: int = Field(default=3600, ge=300, le=7200)


class CacheConfig(BaseModel):
    """Cache configuration model"""
    
    # Redis settings
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    redis_db: int = Field(default=0, ge=0, le=15)
    redis_max_connections: int = Field(default=20, ge=1, le=100)
    
    # Cache settings
    cache_prefix: str = "copywriting:"
    cache_ttl: int = Field(default=7200, ge=60, le=86400)
    enable_compression: bool = True
    compression_threshold: int = Field(default=1024, ge=64, le=1048576)


class ModelConfig(BaseModel):
    """Model configuration model"""
    
    # Model settings
    default_model: str = "gpt2-medium"
    fallback_model: str = "distilgpt2"
    model_cache_dir: str = "./models"
    download_models: bool = True
    
    # Generation settings
    max_tokens: int = Field(default=1024, ge=64, le=4096)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    do_sample: bool = True
    num_return_sequences: int = Field(default=1, ge=1, le=10)
    
    # GPU settings
    enable_gpu: bool = True
    enable_quantization: bool = True
    enable_mixed_precision: bool = True
    gpu_memory_fraction: float = Field(default=0.8, ge=0.1, le=1.0)


class AppConfig(BaseModel):
    """Main application configuration model"""
    
    # Environment
    environment: str = Field(default="development", regex="^(development|staging|production)$")
    debug: bool = False
    version: str = "3.0.0"
    
    # Configurations
    engine: EngineConfig = EngineConfig()
    api: APIConfig = APIConfig()
    security: SecurityConfig = SecurityConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    database: DatabaseConfig = DatabaseConfig()
    cache: CacheConfig = CacheConfig()
    model: ModelConfig = ModelConfig()
    
    # Custom settings
    custom_settings: Dict[str, Any] = {}
    
    @dataclass
class Config:
        validate_assignment = True
        extra = "forbid" 