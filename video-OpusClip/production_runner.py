#!/usr/bin/env python3
"""
Production Runner for Optimized Video-OpusClip

High-performance production deployment with all optimizations enabled.
"""

import asyncio
import uvicorn
import os
import sys
import torch
import gc
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from optimized_api import create_high_performance_api
from optimized_config import get_config, update_config
from optimized_libraries import get_optimized_components, optimize_memory
from performance_monitor import get_performance_monitor
from optimized_cache import get_cache_manager

def setup_production_environment():
    """Setup production environment with all optimizations."""
    print("🚀 Setting up production environment...")
    
    # Environment variables for production
    os.environ.setdefault("MAX_WORKERS", str(os.cpu_count() * 2))
    os.environ.setdefault("BATCH_SIZE", "32")
    os.environ.setdefault("ENABLE_CACHING", "true")
    os.environ.setdefault("USE_GPU", "true")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("ENABLE_STRUCTURED_LOGGING", "true")
    os.environ.setdefault("ENABLE_MIXED_PRECISION", "true")
    os.environ.setdefault("ENABLE_ATTENTION_SLICING", "true")
    os.environ.setdefault("ENABLE_VAE_SLICING", "true")
    os.environ.setdefault("ENABLE_MODEL_CPU_OFFLOAD", "true")

def optimize_for_production():
    """Apply production optimizations."""
    print("⚡ Applying production optimizations...")
    
    config = get_config()
    
    # High-performance settings
    update_config(
        MAX_WORKERS=os.cpu_count() * 2,
        BATCH_SIZE=64,
        ENABLE_PARALLEL_PROCESSING=True,
        ENABLE_ASYNC_PROCESSING=True,
        ENABLE_BATCH_PROCESSING=True,
        ENABLE_MIXED_PRECISION=True
    )
    
    # Cache optimization
    config.performance.cache_max_size = 100000
    config.performance.enable_redis_cache = True
    config.performance.enable_memory_cache = True
    config.performance.enable_compression = True
    
    # API optimization
    config.performance.rate_limit_per_minute = 5000
    config.performance.enable_response_compression = True
    config.performance.max_request_size = 100 * 1024 * 1024  # 100MB
    
    # GPU optimization
    if torch.cuda.is_available():
        config.env.USE_GPU = True
        config.performance.enable_gpu_encoding = True
        config.performance.enable_tensor_cores = True
        config.performance.enable_cudnn_benchmark = True
        
        # Set CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    return config

async def initialize_components():
    """Initialize all optimized components."""
    print("🔧 Initializing optimized components...")
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {device}")
    
    # Initialize components
    components = get_optimized_components(device)
    
    # Initialize cache
    cache_manager = get_cache_manager()
    cache_health = await cache_manager.health_check()
    print(f"💾 Cache health: {cache_health}")
    
    # Initialize performance monitor
    monitor = get_performance_monitor()
    current_metrics = monitor.get_current_metrics()
    print(f"📊 System metrics: CPU {current_metrics.cpu_usage:.1f}%, Memory {current_metrics.memory_usage:.1f}%")
    
    return components

def setup_logging():
    """Setup production logging."""
    import structlog
    
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

def setup_signal_handlers():
    """Setup graceful shutdown handlers."""
    import signal
    
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        optimize_memory()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def health_check():
    """Production health check."""
    print("🏥 Performing production health check...")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU Memory: {gpu_memory:.1f}GB")
    
    # Check cache
    cache_manager = get_cache_manager()
    cache_health = await cache_manager.health_check()
    print(f"💾 Cache: {cache_health}")
    
    # Check performance monitor
    monitor = get_performance_monitor()
    summary = monitor.get_performance_summary()
    print(f"📊 Performance: {summary.get('total_operations', 0)} operations")
    
    # Check memory
    import psutil
    memory = psutil.virtual_memory()
    print(f"🧠 Memory: {memory.percent}% used, {memory.available / 1024**3:.1f}GB available")
    
    print("✅ Health check completed")

def main():
    """Main production runner."""
    print("🚀 Starting Production Video-OpusClip System...")
    
    # Setup environment
    setup_production_environment()
    
    # Setup logging
    setup_logging()
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Optimize configuration
    config = optimize_for_production()
    
    # Initialize components
    components = asyncio.run(initialize_components())
    
    # Health check
    asyncio.run(health_check())
    
    # Create high-performance API
    api = create_high_performance_api()
    
    print(f"🎯 Production configuration:")
    print(f"   - Workers: {config.env.MAX_WORKERS}")
    print(f"   - Batch Size: {config.env.BATCH_SIZE}")
    print(f"   - Cache: {config.env.ENABLE_CACHING}")
    print(f"   - GPU: {config.env.USE_GPU}")
    print(f"   - Rate Limit: {config.performance.rate_limit_per_minute}/min")
    print(f"   - Mixed Precision: {config.env.MIXED_PRECISION}")
    
    # Start production server
    print("🌐 Starting production server...")
    uvicorn.run(
        api.app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for async
        loop="uvloop",
        http="httptools",
        access_log=True,
        log_level="info",
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30
    )

if __name__ == "__main__":
    main() 