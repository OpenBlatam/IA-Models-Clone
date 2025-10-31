from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import multiprocessing as mp
from typing import Optional, List, Dict, Any
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings
        import orjson
        import msgspec
        import uvloop
        import asyncpg
        import redis
        import prometheus_client
        import spacy
        import polars
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Optimized Configuration for Copywriting Service.

High-performance configuration with intelligent defaults and auto-tuning.
"""


class OptimizedCopywritingConfig(BaseSettings):
    """Ultra-optimized configuration with performance tuning."""
    
    # === PERFORMANCE SETTINGS ===
    # Auto-calculated based on system resources
    max_workers: int = Field(
        default_factory=lambda: min(32, mp.cpu_count() * 4),
        description="Maximum worker threads"
    )
    
    max_concurrent_requests: int = Field(
        default_factory=lambda: min(1000, mp.cpu_count() * 100),
        description="Maximum concurrent requests"
    )
    
    # Memory optimization
    max_memory_mb: int = Field(
        default_factory=lambda: min(8192, int(os.environ.get('MEMORY_LIMIT', '4096'))),
        description="Maximum memory usage in MB"
    )
    
    # Connection pooling
    db_pool_size: int = Field(20, description="Database connection pool size")
    redis_pool_size: int = Field(50, description="Redis connection pool size")
    http_pool_size: int = Field(100, description="HTTP connection pool size")
    
    # === CACHING CONFIGURATION ===
    # Multi-level caching
    enable_memory_cache: bool = Field(True, description="Enable in-memory caching")
    enable_redis_cache: bool = Field(True, description="Enable Redis caching")
    enable_disk_cache: bool = Field(False, description="Enable disk caching")
    
    # Cache TTL settings
    cache_ttl_short: int = Field(300, description="Short cache TTL (5 min)")
    cache_ttl_medium: int = Field(1800, description="Medium cache TTL (30 min)")
    cache_ttl_long: int = Field(3600, description="Long cache TTL (1 hour)")
    
    # Cache sizes
    memory_cache_size: int = Field(1000, description="Memory cache max items")
    disk_cache_size_mb: int = Field(500, description="Disk cache size in MB")
    
    # === SECURITY SETTINGS ===
    api_key: str = Field("optimized-copywriting-key", description="API authentication key")
    allowed_origins: List[str] = Field(["*"], description="CORS allowed origins")
    rate_limit_per_minute: int = Field(100, description="Rate limit per minute per IP")
    
    # === AI/MODEL SETTINGS ===
    default_model: str = Field("optimized-local", description="Default AI model")
    max_tokens: int = Field(2000, description="Maximum tokens per request")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Model temperature")
    max_variants: int = Field(5, ge=1, le=20, description="Maximum variants per request")
    
    # Model performance settings
    enable_model_caching: bool = Field(True, description="Enable model response caching")
    model_timeout: int = Field(30, description="Model request timeout in seconds")
    enable_parallel_generation: bool = Field(True, description="Enable parallel variant generation")
    
    # === DATABASE SETTINGS ===
    database_url: str = Field(
        "sqlite:///copywriting_optimized.db",
        description="Database connection URL"
    )
    enable_async_db: bool = Field(True, description="Enable async database operations")
    db_echo: bool = Field(False, description="Enable SQL query logging")
    
    # === REDIS SETTINGS ===
    redis_url: str = Field("redis://localhost:6379/0", description="Redis connection URL")
    redis_password: Optional[str] = Field(None, description="Redis password")
    redis_ssl: bool = Field(False, description="Enable Redis SSL")
    
    # === MONITORING & LOGGING ===
    enable_metrics: bool = Field(True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(True, description="Enable request tracing")
    log_level: str = Field("INFO", description="Logging level")
    enable_structured_logging: bool = Field(True, description="Enable structured logging")
    
    # Performance monitoring
    enable_profiling: bool = Field(False, description="Enable performance profiling")
    metrics_port: int = Field(9090, description="Metrics server port")
    health_check_interval: int = Field(30, description="Health check interval in seconds")
    
    # === OPTIMIZATION FLAGS ===
    # JSON optimization
    use_orjson: bool = Field(True, description="Use orjson for JSON operations")
    use_msgspec: bool = Field(True, description="Use msgspec for serialization")
    
    # Async optimization
    use_uvloop: bool = Field(True, description="Use uvloop event loop (Unix only)")
    enable_async_io: bool = Field(True, description="Enable async I/O operations")
    
    # Text processing optimization
    enable_fast_text_processing: bool = Field(True, description="Enable fast text processing")
    use_spacy_gpu: bool = Field(False, description="Use spaCy GPU acceleration")
    
    # === EXTERNAL SERVICES ===
    # AI APIs (optional)
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    
    # Analytics
    enable_analytics: bool = Field(True, description="Enable analytics collection")
    analytics_batch_size: int = Field(100, description="Analytics batch size")
    
    # === PLATFORM SPECIFIC SETTINGS ===
    platform_configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "instagram": {"max_chars": 2200, "max_hashtags": 30},
            "twitter": {"max_chars": 280, "max_hashtags": 10},
            "facebook": {"max_chars": 63206, "max_hashtags": 15},
            "linkedin": {"max_chars": 3000, "max_hashtags": 5},
            "tiktok": {"max_chars": 300, "max_hashtags": 20},
        },
        description="Platform-specific configurations"
    )
    
    # === DEVELOPMENT SETTINGS ===
    debug: bool = Field(False, description="Enable debug mode")
    reload: bool = Field(False, description="Enable auto-reload")
    
    @dataclass
class Config:
        env_prefix = "COPYWRITING_"
        env_file = ".env"
        case_sensitive = False
        
    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """Get configuration for specific platform."""
        return self.platform_configs.get(platform, {
            "max_chars": 1000,
            "max_hashtags": 10
        })
    
    def is_high_performance_mode(self) -> bool:
        """Check if running in high-performance mode."""
        return (
            self.max_workers >= 16 and
            self.enable_memory_cache and
            self.enable_redis_cache and
            self.use_orjson and
            self.enable_parallel_generation
        )
    
    def get_optimization_level(self) -> str:
        """Get current optimization level."""
        if self.is_high_performance_mode():
            return "ULTRA"
        elif self.max_workers >= 8 and self.enable_memory_cache:
            return "HIGH"
        elif self.max_workers >= 4:
            return "MEDIUM"
        else:
            return "BASIC"

@lru_cache(maxsize=1)
def get_config() -> OptimizedCopywritingConfig:
    """Get cached configuration instance."""
    return OptimizedCopywritingConfig()

# Auto-detect optimization capabilities
def detect_optimization_capabilities() -> Dict[str, bool]:
    """Detect available optimization libraries and capabilities."""
    capabilities = {}
    
    # JSON libraries
    try:
        capabilities["orjson"] = True
    except ImportError:
        capabilities["orjson"] = False
    
    try:
        capabilities["msgspec"] = True
    except ImportError:
        capabilities["msgspec"] = False
    
    # Async libraries
    try:
        capabilities["uvloop"] = True
    except ImportError:
        capabilities["uvloop"] = False
    
    # Database drivers
    try:
        capabilities["asyncpg"] = True
    except ImportError:
        capabilities["asyncpg"] = False
    
    # Caching
    try:
        capabilities["redis"] = True
    except ImportError:
        capabilities["redis"] = False
    
    # Performance monitoring
    try:
        capabilities["prometheus"] = True
    except ImportError:
        capabilities["prometheus"] = False
    
    # Text processing
    try:
        capabilities["spacy"] = True
    except ImportError:
        capabilities["spacy"] = False
    
    try:
        capabilities["polars"] = True
    except ImportError:
        capabilities["polars"] = False
    
    return capabilities

def get_performance_recommendations() -> List[str]:
    """Get performance optimization recommendations."""
    config = get_config()
    capabilities = detect_optimization_capabilities()
    recommendations = []
    
    # Check for missing high-impact libraries
    if not capabilities.get("orjson"):
        recommendations.append("Install orjson for 5x faster JSON operations: pip install orjson")
    
    if not capabilities.get("msgspec"):
        recommendations.append("Install msgspec for ultra-fast serialization: pip install msgspec")
    
    if not capabilities.get("uvloop") and os.name != 'nt':
        recommendations.append("Install uvloop for 4x faster async operations: pip install uvloop")
    
    if not capabilities.get("redis") and config.enable_redis_cache:
        recommendations.append("Install redis for caching: pip install redis")
    
    if not capabilities.get("polars"):
        recommendations.append("Install polars for 20x faster data processing: pip install polars")
    
    # Configuration recommendations
    if config.max_workers < 8:
        recommendations.append(f"Increase max_workers to {mp.cpu_count() * 2} for better performance")
    
    if not config.enable_memory_cache:
        recommendations.append("Enable memory caching for faster response times")
    
    if not config.enable_parallel_generation:
        recommendations.append("Enable parallel generation for faster variant creation")
    
    return recommendations

# Export configuration
__all__ = [
    "OptimizedCopywritingConfig",
    "get_config", 
    "detect_optimization_capabilities",
    "get_performance_recommendations"
] 