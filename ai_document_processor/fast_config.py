"""
Fast Configuration - Optimized Settings for Maximum Speed
========================================================

Configuration optimized for ultra-high performance document processing.
"""

import os
from typing import Dict, List, Optional
from pydantic import BaseSettings, Field
import multiprocessing as mp

class FastSettings(BaseSettings):
    """Optimized settings for maximum speed"""
    
    # Application settings
    app_name: str = "Fast AI Document Processor"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8001, env="PORT")
    
    # Performance optimizations
    max_workers: int = Field(default=None, env="MAX_WORKERS")
    chunk_size: int = Field(default=8192, env="CHUNK_SIZE")
    enable_streaming: bool = Field(default=True, env="ENABLE_STREAMING")
    enable_parallel_ai: bool = Field(default=True, env="ENABLE_PARALLEL_AI")
    enable_compression: bool = Field(default=True, env="ENABLE_COMPRESSION")
    
    # Cache settings for speed
    cache_max_memory_mb: int = Field(default=1024, env="CACHE_MAX_MEMORY_MB")
    cache_default_ttl: int = Field(default=3600, env="CACHE_DEFAULT_TTL")
    cache_redis_url: Optional[str] = Field(default=None, env="CACHE_REDIS_URL")
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    
    # AI settings optimized for speed
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=2000, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.3, env="OPENAI_TEMPERATURE")  # Lower for faster responses
    openai_timeout: int = Field(default=30, env="OPENAI_TIMEOUT")
    
    # File processing limits
    max_file_size: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    max_batch_size: int = Field(default=10, env="MAX_BATCH_SIZE")
    allowed_extensions: List[str] = Field(
        default=[".md", ".pdf", ".docx", ".doc", ".txt"], 
        env="ALLOWED_EXTENSIONS"
    )
    temp_dir: str = Field(default="/tmp", env="TEMP_DIR")
    
    # Processing optimizations
    max_text_length: int = Field(default=200000, env="MAX_TEXT_LENGTH")  # Increased for better processing
    min_confidence_threshold: float = Field(default=0.3, env="MIN_CONFIDENCE_THRESHOLD")  # Lower for speed
    enable_ai_classification: bool = Field(default=True, env="ENABLE_AI_CLASSIFICATION")
    enable_ml_classification: bool = Field(default=False, env="ENABLE_ML_CLASSIFICATION")
    
    # Memory optimization
    memory_optimization: bool = Field(default=True, env="MEMORY_OPTIMIZATION")
    gc_threshold: int = Field(default=100, env="GC_THRESHOLD")  # Garbage collection threshold
    max_memory_usage_percent: float = Field(default=80.0, env="MAX_MEMORY_USAGE_PERCENT")
    
    # Monitoring and metrics
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_retention_hours: int = Field(default=24, env="METRICS_RETENTION_HOURS")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")  # Faster health checks
    
    # Logging optimized for performance
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    enable_structured_logging: bool = Field(default=True, env="ENABLE_STRUCTURED_LOGGING")
    
    # Security settings
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    
    # Database settings (optional)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    enable_database_caching: bool = Field(default=False, env="ENABLE_DATABASE_CACHING")
    
    # Advanced optimizations
    enable_uvloop: bool = Field(default=True, env="ENABLE_UVLOOP")
    enable_gzip: bool = Field(default=True, env="ENABLE_GZIP")
    gzip_minimum_size: int = Field(default=1000, env="GZIP_MINIMUM_SIZE")
    
    # Batch processing optimizations
    batch_processing_timeout: int = Field(default=300, env="BATCH_PROCESSING_TIMEOUT")  # 5 minutes
    batch_parallel_limit: int = Field(default=None, env="BATCH_PARALLEL_LIMIT")
    
    # AI processing optimizations
    ai_processing_timeout: int = Field(default=60, env="AI_PROCESSING_TIMEOUT")
    ai_retry_attempts: int = Field(default=3, env="AI_RETRY_ATTEMPTS")
    ai_retry_delay: float = Field(default=1.0, env="AI_RETRY_DELAY")
    
    # File handling optimizations
    file_buffer_size: int = Field(default=65536, env="FILE_BUFFER_SIZE")  # 64KB buffer
    enable_file_streaming: bool = Field(default=True, env="ENABLE_FILE_STREAMING")
    max_concurrent_files: int = Field(default=50, env="MAX_CONCURRENT_FILES")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Auto-configure max_workers if not set
        if self.max_workers is None:
            cpu_count = mp.cpu_count() or 1
            self.max_workers = min(32, cpu_count * 2 + 4)
        
        # Auto-configure batch_parallel_limit if not set
        if self.batch_parallel_limit is None:
            self.batch_parallel_limit = min(self.max_workers, 10)
        
        # Validate settings
        self._validate_settings()
    
    def _validate_settings(self):
        """Validate configuration settings"""
        if self.max_file_size <= 0:
            raise ValueError("max_file_size must be positive")
        
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.cache_max_memory_mb <= 0:
            raise ValueError("cache_max_memory_mb must be positive")
        
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        
        if not self.allowed_extensions:
            raise ValueError("allowed_extensions cannot be empty")
    
    def get_optimized_settings(self) -> Dict[str, any]:
        """Get optimized settings for different components"""
        return {
            'fastapi': {
                'host': self.host,
                'port': self.port,
                'debug': self.debug,
                'workers': 1,  # Single worker for async processing
                'loop': 'asyncio' if not self.enable_uvloop else 'uvloop',
                'access_log': True,
                'log_level': self.log_level.lower()
            },
            'cache': {
                'max_memory_mb': self.cache_max_memory_mb,
                'default_ttl': self.cache_default_ttl,
                'redis_url': self.cache_redis_url,
                'enable_compression': self.enable_compression
            },
            'processor': {
                'max_workers': self.max_workers,
                'chunk_size': self.chunk_size,
                'enable_streaming': self.enable_streaming,
                'enable_parallel_ai': self.enable_parallel_ai,
                'memory_optimization': self.memory_optimization
            },
            'monitoring': {
                'enable_metrics': self.enable_metrics,
                'retention_hours': self.metrics_retention_hours,
                'health_check_interval': self.health_check_interval
            },
            'ai': {
                'model': self.openai_model,
                'max_tokens': self.openai_max_tokens,
                'temperature': self.openai_temperature,
                'timeout': self.openai_timeout,
                'retry_attempts': self.ai_retry_attempts,
                'retry_delay': self.ai_retry_delay
            },
            'files': {
                'max_size': self.max_file_size,
                'max_batch_size': self.max_batch_size,
                'allowed_extensions': self.allowed_extensions,
                'buffer_size': self.file_buffer_size,
                'enable_streaming': self.enable_file_streaming,
                'max_concurrent': self.max_concurrent_files
            }
        }
    
    def get_performance_tips(self) -> List[str]:
        """Get performance optimization tips"""
        tips = []
        
        if self.cache_redis_url:
            tips.append("✅ Redis cache enabled for maximum performance")
        else:
            tips.append("⚠️ Consider enabling Redis cache for better performance")
        
        if self.enable_uvloop:
            tips.append("✅ UVLoop enabled for faster async processing")
        else:
            tips.append("⚠️ Consider enabling UVLoop for better async performance")
        
        if self.enable_streaming:
            tips.append("✅ Streaming enabled for large file processing")
        
        if self.enable_parallel_ai:
            tips.append("✅ Parallel AI processing enabled")
        
        if self.memory_optimization:
            tips.append("✅ Memory optimization enabled")
        
        if self.max_workers >= 8:
            tips.append("✅ High worker count configured for parallel processing")
        else:
            tips.append("⚠️ Consider increasing max_workers for better parallel processing")
        
        if self.cache_max_memory_mb >= 512:
            tips.append("✅ Large cache size configured")
        else:
            tips.append("⚠️ Consider increasing cache size for better performance")
        
        return tips

# Global settings instance
settings = FastSettings()

# Performance optimization presets
PERFORMANCE_PRESETS = {
    'ultra_fast': {
        'max_workers': 32,
        'chunk_size': 16384,
        'cache_max_memory_mb': 2048,
        'enable_streaming': True,
        'enable_parallel_ai': True,
        'enable_compression': True,
        'openai_temperature': 0.1,
        'openai_timeout': 15,
        'memory_optimization': True,
        'enable_uvloop': True
    },
    'balanced': {
        'max_workers': 16,
        'chunk_size': 8192,
        'cache_max_memory_mb': 1024,
        'enable_streaming': True,
        'enable_parallel_ai': True,
        'enable_compression': True,
        'openai_temperature': 0.3,
        'openai_timeout': 30,
        'memory_optimization': True,
        'enable_uvloop': True
    },
    'memory_efficient': {
        'max_workers': 8,
        'chunk_size': 4096,
        'cache_max_memory_mb': 512,
        'enable_streaming': True,
        'enable_parallel_ai': False,
        'enable_compression': True,
        'openai_temperature': 0.5,
        'openai_timeout': 45,
        'memory_optimization': True,
        'enable_uvloop': False
    }
}

def apply_performance_preset(preset_name: str) -> FastSettings:
    """Apply a performance preset"""
    if preset_name not in PERFORMANCE_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PERFORMANCE_PRESETS.keys())}")
    
    preset = PERFORMANCE_PRESETS[preset_name]
    return FastSettings(**preset)

def get_system_recommendations() -> Dict[str, any]:
    """Get system-specific performance recommendations"""
    import psutil
    
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    recommendations = {
        'cpu_count': cpu_count,
        'memory_gb': round(memory_gb, 2),
        'recommended_workers': min(32, cpu_count * 2 + 4),
        'recommended_cache_mb': min(2048, int(memory_gb * 0.25 * 1024)),
        'preset': 'ultra_fast' if memory_gb >= 8 and cpu_count >= 8 else 'balanced'
    }
    
    if memory_gb < 4:
        recommendations['preset'] = 'memory_efficient'
        recommendations['warnings'] = ['Low memory detected, using memory-efficient preset']
    
    return recommendations

















