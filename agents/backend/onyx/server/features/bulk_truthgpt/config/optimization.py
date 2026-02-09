"""
Optimization Configuration
========================

Configuration settings for performance optimization.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

class OptimizationLevel(str, Enum):
    """Optimization levels."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    AGGRESSIVE = "aggressive"

class CacheStrategy(str, Enum):
    """Cache strategies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    SIZE = "size"

class CompressionLevel(str, Enum):
    """Compression levels."""
    FAST = "fast"
    BALANCED = "balanced"
    MAXIMUM = "maximum"

class OptimizationConfig(BaseModel):
    """Optimization configuration."""
    
    # Performance optimization
    performance_level: OptimizationLevel = OptimizationLevel.ADVANCED
    memory_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    cpu_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    gc_threshold: int = Field(default=1000, ge=100)
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_gc_optimization: bool = True
    enable_io_optimization: bool = True
    enable_network_optimization: bool = True
    
    # Cache configuration
    cache_enabled: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.LRU
    l1_cache_size: int = Field(default=10000, ge=1000)
    l2_cache_enabled: bool = True
    l3_cache_enabled: bool = True
    cache_ttl: int = Field(default=3600, ge=60)
    cache_compression: bool = True
    
    # Batch processing
    batch_processing_enabled: bool = True
    max_batch_size: int = Field(default=100, ge=10)
    max_workers: int = Field(default=4, ge=1)
    processing_timeout: int = Field(default=300, ge=30)
    retry_attempts: int = Field(default=3, ge=1)
    retry_delay: float = Field(default=1.0, ge=0.1)
    
    # Compression
    compression_enabled: bool = True
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    compression_algorithm: str = "gzip"
    compression_threshold: int = Field(default=1024, ge=100)
    
    # Lazy loading
    lazy_loading_enabled: bool = True
    preload_count: int = Field(default=10, ge=1)
    lazy_ttl: int = Field(default=3600, ge=60)
    
    # Monitoring
    monitoring_enabled: bool = True
    metrics_interval: int = Field(default=30, ge=5)
    performance_logging: bool = True
    
    # Database optimization
    db_pool_size: int = Field(default=20, ge=5)
    db_max_overflow: int = Field(default=30, ge=10)
    db_pool_timeout: int = Field(default=30, ge=5)
    db_pool_recycle: int = Field(default=3600, ge=300)
    
    # Redis optimization
    redis_pool_size: int = Field(default=20, ge=5)
    redis_max_connections: int = Field(default=100, ge=10)
    redis_socket_timeout: int = Field(default=5, ge=1)
    redis_socket_connect_timeout: int = Field(default=5, ge=1)
    
    # Network optimization
    http_timeout: int = Field(default=30, ge=5)
    http_retries: int = Field(default=3, ge=1)
    http_backoff_factor: float = Field(default=0.3, ge=0.1)
    connection_pool_size: int = Field(default=100, ge=10)
    
    # Memory optimization
    memory_pool_size: int = Field(default=1000, ge=100)
    memory_cleanup_interval: int = Field(default=300, ge=60)
    memory_leak_detection: bool = True
    memory_profiling: bool = False
    
    # CPU optimization
    cpu_affinity: bool = False
    thread_pool_size: int = Field(default=4, ge=1)
    process_pool_size: int = Field(default=2, ge=1)
    cpu_monitoring: bool = True
    
    # I/O optimization
    io_buffer_size: int = Field(default=8192, ge=1024)
    io_timeout: int = Field(default=30, ge=5)
    io_retries: int = Field(default=3, ge=1)
    async_io: bool = True
    
    # Security optimization
    encryption_enabled: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_interval: int = Field(default=86400, ge=3600)
    
    # Logging optimization
    log_level: str = "INFO"
    log_rotation: bool = True
    log_compression: bool = True
    log_retention_days: int = Field(default=30, ge=1)
    
    # Development settings
    debug_mode: bool = False
    profiling_enabled: bool = False
    hot_reload: bool = False
    
    # Production settings
    production_mode: bool = False
    health_check_interval: int = Field(default=30, ge=5)
    graceful_shutdown_timeout: int = Field(default=30, ge=5)
    
    class Config:
        env_prefix = "BULK_TRUTHGPT_OPTIMIZATION_"
        case_sensitive = False

# Default optimization configuration
default_optimization_config = OptimizationConfig()

# Production optimization configuration
production_optimization_config = OptimizationConfig(
    performance_level=OptimizationLevel.AGGRESSIVE,
    memory_threshold=0.9,
    cpu_threshold=0.9,
    cache_strategy=CacheStrategy.LRU,
    l1_cache_size=50000,
    batch_processing_enabled=True,
    max_batch_size=500,
    max_workers=8,
    compression_enabled=True,
    compression_level=CompressionLevel.MAXIMUM,
    lazy_loading_enabled=True,
    preload_count=50,
    monitoring_enabled=True,
    metrics_interval=10,
    db_pool_size=50,
    redis_pool_size=50,
    memory_pool_size=5000,
    thread_pool_size=8,
    process_pool_size=4,
    production_mode=True
)

# Development optimization configuration
development_optimization_config = OptimizationConfig(
    performance_level=OptimizationLevel.BASIC,
    memory_threshold=0.7,
    cpu_threshold=0.7,
    cache_strategy=CacheStrategy.TTL,
    l1_cache_size=5000,
    batch_processing_enabled=True,
    max_batch_size=50,
    max_workers=2,
    compression_enabled=False,
    lazy_loading_enabled=True,
    preload_count=5,
    monitoring_enabled=True,
    metrics_interval=60,
    db_pool_size=10,
    redis_pool_size=10,
    memory_pool_size=1000,
    thread_pool_size=2,
    process_pool_size=1,
    debug_mode=True,
    profiling_enabled=True,
    hot_reload=True,
    production_mode=False
)

# Test optimization configuration
test_optimization_config = OptimizationConfig(
    performance_level=OptimizationLevel.NONE,
    memory_threshold=0.5,
    cpu_threshold=0.5,
    cache_enabled=False,
    batch_processing_enabled=False,
    compression_enabled=False,
    lazy_loading_enabled=False,
    monitoring_enabled=False,
    db_pool_size=5,
    redis_pool_size=5,
    memory_pool_size=100,
    thread_pool_size=1,
    process_pool_size=1,
    debug_mode=True,
    production_mode=False
)

def get_optimization_config(environment: str = "development") -> OptimizationConfig:
    """Get optimization configuration for environment."""
    configs = {
        "development": development_optimization_config,
        "production": production_optimization_config,
        "test": test_optimization_config
    }
    
    return configs.get(environment, default_optimization_config)

def validate_optimization_config(config: OptimizationConfig) -> Dict[str, Any]:
    """Validate optimization configuration."""
    issues = []
    
    # Check memory thresholds
    if config.memory_threshold > 0.95:
        issues.append("Memory threshold too high, may cause system instability")
    
    if config.cpu_threshold > 0.95:
        issues.append("CPU threshold too high, may cause system instability")
    
    # Check cache sizes
    if config.l1_cache_size > 100000:
        issues.append("L1 cache size too large, may cause memory issues")
    
    # Check batch processing
    if config.max_batch_size > 1000:
        issues.append("Max batch size too large, may cause performance issues")
    
    if config.max_workers > 16:
        issues.append("Max workers too high, may cause resource contention")
    
    # Check database pool
    if config.db_pool_size > 100:
        issues.append("Database pool size too large, may cause connection issues")
    
    # Check Redis pool
    if config.redis_pool_size > 100:
        issues.append("Redis pool size too large, may cause connection issues")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": []
    }











