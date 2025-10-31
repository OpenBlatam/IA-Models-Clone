"""
Optimized Configuration System for Video-OpusClip

Enhanced configuration with environment-based settings, performance tuning,
and intelligent resource management.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import structlog

logger = structlog.get_logger()

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

@dataclass
class EnvironmentConfig:
    """Environment-based configuration with intelligent defaults."""
    
    # API Configuration
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    COHERE_API_KEY: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    
    # Performance Configuration
    MAX_WORKERS: int = field(default_factory=lambda: int(os.getenv("MAX_WORKERS", str(os.cpu_count() or 4))))
    BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "10")))
    CHUNK_SIZE: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    TIMEOUT: float = field(default_factory=lambda: float(os.getenv("TIMEOUT", "30.0")))
    
    # Memory Configuration
    MAX_MEMORY_MB: int = field(default_factory=lambda: int(os.getenv("MAX_MEMORY_MB", "8192")))
    CACHE_SIZE: int = field(default_factory=lambda: int(os.getenv("CACHE_SIZE", "1000")))
    ENABLE_CACHING: bool = field(default_factory=lambda: os.getenv("ENABLE_CACHING", "true").lower() == "true")
    
    # GPU Configuration
    USE_GPU: bool = field(default_factory=lambda: os.getenv("USE_GPU", "true").lower() == "true")
    GPU_MEMORY_LIMIT: float = field(default_factory=lambda: float(os.getenv("GPU_MEMORY_LIMIT", "0.8")))
    MIXED_PRECISION: bool = field(default_factory=lambda: os.getenv("MIXED_PRECISION", "true").lower() == "true")
    
    # Database Configuration
    DATABASE_URL: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///video_cache.db"))
    REDIS_URL: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    
    # Logging Configuration
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    ENABLE_STRUCTURED_LOGGING: bool = field(default_factory=lambda: os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true")
    
    # Feature Flags
    ENABLE_LANGCHAIN: bool = field(default_factory=lambda: os.getenv("ENABLE_LANGCHAIN", "true").lower() == "true")
    ENABLE_VIRAL_ANALYSIS: bool = field(default_factory=lambda: os.getenv("ENABLE_VIRAL_ANALYSIS", "true").lower() == "true")
    ENABLE_BATCH_PROCESSING: bool = field(default_factory=lambda: os.getenv("ENABLE_BATCH_PROCESSING", "true").lower() == "true")
    ENABLE_PARALLEL_PROCESSING: bool = field(default_factory=lambda: os.getenv("ENABLE_PARALLEL_PROCESSING", "true").lower() == "true")

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

@dataclass
class PerformanceConfig:
    """Optimized performance configuration."""
    
    # Parallel Processing
    parallel_backend: str = "auto"  # auto, thread, process, joblib, ray
    max_concurrent_tasks: int = 100
    task_timeout: float = 60.0
    enable_uvloop: bool = True
    enable_numba: bool = True
    
    # Memory Management
    enable_memory_pooling: bool = True
    memory_pool_size: int = 1000
    enable_object_reuse: bool = True
    gc_threshold: int = 1000
    
    # Caching Strategy
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 10000
    enable_redis_cache: bool = True
    enable_memory_cache: bool = True
    
    # GPU Optimization
    enable_tensor_cores: bool = True
    enable_cudnn_benchmark: bool = True
    enable_mixed_precision: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Batch Processing
    optimal_batch_size: int = 16
    dynamic_batch_sizing: bool = True
    batch_timeout: float = 300.0
    
    # API Optimization
    enable_response_compression: bool = True
    enable_request_validation: bool = True
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    rate_limit_per_minute: int = 1000

# =============================================================================
# LANGCHAIN OPTIMIZATION
# =============================================================================

@dataclass
class LangChainOptimizedConfig:
    """Optimized LangChain configuration for better performance."""
    
    # Model Configuration
    model_name: str = "gpt-4-turbo-preview"
    temperature: float = 0.3  # Lower for more consistent results
    max_tokens: int = 1000  # Reduced for faster responses
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Performance Settings
    enable_streaming: bool = False  # Disabled for batch processing
    enable_caching: bool = True
    cache_ttl: int = 3600
    max_retries: int = 2  # Reduced for faster failure
    retry_delay: float = 1.0
    
    # Batch Processing
    batch_size: int = 10
    concurrent_requests: int = 5
    request_timeout: float = 30.0
    
    # Memory Optimization
    enable_model_reuse: bool = True
    enable_prompt_caching: bool = True
    max_prompt_cache_size: int = 1000
    
    # Feature Flags
    enable_content_analysis: bool = True
    enable_viral_analysis: bool = True
    enable_engagement_analysis: bool = True
    enable_audience_analysis: bool = True
    enable_title_optimization: bool = True
    enable_caption_optimization: bool = True
    enable_timing_optimization: bool = True

# =============================================================================
# VIDEO PROCESSING OPTIMIZATION
# =============================================================================

@dataclass
class VideoProcessingOptimizedConfig:
    """Optimized video processing configuration."""
    
    # Video Processing
    max_video_duration: float = 600.0  # 10 minutes
    target_fps: int = 30
    target_resolution: str = "1080p"
    enable_hardware_acceleration: bool = True
    codec: str = "h264"
    bitrate: str = "2M"
    
    # Clip Generation
    min_clip_length: float = 5.0
    max_clip_length: float = 60.0
    optimal_clip_length: float = 30.0
    enable_smart_cutting: bool = True
    enable_audio_analysis: bool = True
    
    # Quality Settings
    enable_quality_optimization: bool = True
    quality_preset: str = "fast"  # fast, balanced, quality
    enable_noise_reduction: bool = True
    enable_stabilization: bool = False  # Disabled for performance
    
    # Parallel Processing
    max_concurrent_videos: int = 4
    enable_gpu_encoding: bool = True
    enable_parallel_encoding: bool = True

# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

@dataclass
class OptimizedConfig:
    """Main optimized configuration class."""
    
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    langchain: LangChainOptimizedConfig = field(default_factory=LangChainOptimizedConfig)
    video: VideoProcessingOptimizedConfig = field(default_factory=VideoProcessingOptimizedConfig)
    
    def __post_init__(self):
        """Post-initialization validation and optimization."""
        self._validate_config()
        self._optimize_settings()
        self._setup_logging()
    
    def _validate_config(self):
        """Validate configuration settings."""
        if self.env.MAX_WORKERS <= 0:
            self.env.MAX_WORKERS = os.cpu_count() or 4
        
        if self.env.BATCH_SIZE <= 0:
            self.env.BATCH_SIZE = 10
        
        if self.performance.max_concurrent_tasks <= 0:
            self.performance.max_concurrent_tasks = 100
    
    def _optimize_settings(self):
        """Optimize settings based on system capabilities."""
        # Auto-detect GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                self.env.USE_GPU = True
                self.performance.enable_gpu_encoding = True
                logger.info("GPU detected, enabling GPU acceleration")
            else:
                self.env.USE_GPU = False
                self.performance.enable_gpu_encoding = False
                logger.info("No GPU detected, using CPU processing")
        except ImportError:
            self.env.USE_GPU = False
            self.performance.enable_gpu_encoding = False
            logger.warning("PyTorch not available, using CPU processing")
        
        # Optimize batch size based on available memory
        available_memory = self._get_available_memory()
        if available_memory < 4096:  # Less than 4GB
            self.env.BATCH_SIZE = 5
            self.performance.optimal_batch_size = 8
            logger.info("Low memory detected, reducing batch sizes")
        elif available_memory > 16384:  # More than 16GB
            self.env.BATCH_SIZE = 20
            self.performance.optimal_batch_size = 32
            logger.info("High memory detected, increasing batch sizes")
    
    def _get_available_memory(self) -> int:
        """Get available system memory in MB."""
        try:
            import psutil
            return psutil.virtual_memory().available // (1024 * 1024)
        except ImportError:
            return 8192  # Default to 8GB
    
    def _setup_logging(self):
        """Setup optimized logging configuration."""
        if self.env.ENABLE_STRUCTURED_LOGGING:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "env": self.env.__dict__,
            "performance": self.performance.__dict__,
            "langchain": self.langchain.__dict__,
            "video": self.video.__dict__
        }
    
    def save_to_file(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'OptimizedConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        config.env = EnvironmentConfig(**config_dict.get("env", {}))
        config.performance = PerformanceConfig(**config_dict.get("performance", {}))
        config.langchain = LangChainOptimizedConfig(**config_dict.get("langchain", {}))
        config.video = VideoProcessingOptimizedConfig(**config_dict.get("video", {}))
        
        return config

# =============================================================================
# CONFIGURATION INSTANCE
# =============================================================================

# Global configuration instance
config = OptimizedConfig()

def get_config() -> OptimizedConfig:
    """Get the global configuration instance."""
    return config

def update_config(**kwargs):
    """Update configuration with new values."""
    global config
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.env, key):
            setattr(config.env, key, value)
        elif hasattr(config.performance, key):
            setattr(config.performance, key, value)
        elif hasattr(config.langchain, key):
            setattr(config.langchain, key, value)
        elif hasattr(config.video, key):
            setattr(config.video, key, value)
        else:
            logger.warning(f"Unknown configuration key: {key}")
    
    # Re-validate and optimize
    config._validate_config()
    config._optimize_settings() 