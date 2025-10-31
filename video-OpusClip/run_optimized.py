#!/usr/bin/env python3
"""
Optimized Video-OpusClip Startup Script

High-performance startup with automatic optimization and monitoring.
"""

import asyncio
import uvicorn
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from optimized_api import get_app, create_high_performance_api
from optimized_config import get_config, update_config
from performance_monitor import get_performance_monitor
from optimized_cache import get_cache_manager

def setup_environment():
    """Setup optimized environment variables."""
    os.environ.setdefault("MAX_WORKERS", str(os.cpu_count() or 4))
    os.environ.setdefault("BATCH_SIZE", "16")
    os.environ.setdefault("ENABLE_CACHING", "true")
    os.environ.setdefault("USE_GPU", "true")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("ENABLE_STRUCTURED_LOGGING", "true")

def optimize_config():
    """Apply performance optimizations."""
    config = get_config()
    
    # High-performance settings
    update_config(
        MAX_WORKERS=os.cpu_count() * 2,
        BATCH_SIZE=32,
        ENABLE_PARALLEL_PROCESSING=True,
        ENABLE_ASYNC_PROCESSING=True,
        ENABLE_BATCH_PROCESSING=True
    )
    
    # Cache optimization
    config.performance.cache_max_size = 50000
    config.performance.enable_redis_cache = True
    config.performance.enable_memory_cache = True
    
    # API optimization
    config.performance.rate_limit_per_minute = 2000
    config.performance.enable_response_compression = True
    
    return config

async def startup_checks():
    """Perform startup health checks."""
    print("🔍 Performing startup checks...")
    
    # Check cache
    cache_manager = get_cache_manager()
    cache_health = await cache_manager.health_check()
    print(f"✅ Cache health: {cache_health}")
    
    # Check performance monitor
    monitor = get_performance_monitor()
    current_metrics = monitor.get_current_metrics()
    print(f"✅ System metrics: CPU {current_metrics.cpu_usage:.1f}%, Memory {current_metrics.memory_usage:.1f}%")
    
    # Check configuration
    config = get_config()
    print(f"✅ Configuration loaded: {config.env.MAX_WORKERS} workers, {config.env.BATCH_SIZE} batch size")

def main():
    """Main startup function."""
    print("🚀 Starting Optimized Video-OpusClip System...")
    
    # Setup environment
    setup_environment()
    
    # Optimize configuration
    config = optimize_config()
    
    # Create high-performance API
    api = create_high_performance_api()
    
    # Run startup checks
    asyncio.run(startup_checks())
    
    print(f"🎯 Optimized configuration:")
    print(f"   - Workers: {config.env.MAX_WORKERS}")
    print(f"   - Batch Size: {config.env.BATCH_SIZE}")
    print(f"   - Cache: {config.env.ENABLE_CACHING}")
    print(f"   - GPU: {config.env.USE_GPU}")
    print(f"   - Rate Limit: {config.performance.rate_limit_per_minute}/min")
    
    # Start server
    uvicorn.run(
        api.app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for async
        loop="uvloop",
        http="httptools",
        access_log=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 