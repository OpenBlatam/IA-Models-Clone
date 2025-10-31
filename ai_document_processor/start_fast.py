#!/usr/bin/env python3
"""
Fast Startup Script - Optimized for Maximum Speed
================================================

Startup script with performance optimizations and system tuning.
"""

import asyncio
import logging
import os
import sys
import signal
import time
from pathlib import Path
from typing import Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fast_config import settings, get_system_recommendations, apply_performance_preset
import uvicorn

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fast_processor.log') if not settings.debug else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_performance_optimizations():
    """Setup system-level performance optimizations"""
    try:
        # Enable UVLoop for faster async processing
        if settings.enable_uvloop:
            try:
                import uvloop
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                logger.info("✅ UVLoop enabled for faster async processing")
            except ImportError:
                logger.warning("⚠️ UVLoop not available, using default event loop")
        
        # Set optimal thread pool size
        if hasattr(asyncio, 'set_default_executor'):
            import concurrent.futures
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=settings.max_workers
            )
            asyncio.set_default_executor(executor)
            logger.info(f"✅ Thread pool executor configured with {settings.max_workers} workers")
        
        # Optimize garbage collection
        import gc
        gc.set_threshold(settings.gc_threshold)
        logger.info(f"✅ Garbage collection threshold set to {settings.gc_threshold}")
        
        # Set process priority (Unix only)
        if os.name != 'nt':
            try:
                import psutil
                process = psutil.Process()
                process.nice(-5)  # Higher priority
                logger.info("✅ Process priority increased")
            except (ImportError, PermissionError):
                logger.warning("⚠️ Could not set process priority")
        
        logger.info("🚀 Performance optimizations applied successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to apply performance optimizations: {e}")

def check_system_requirements():
    """Check system requirements and provide recommendations"""
    try:
        import psutil
        
        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        
        # Check disk space
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        
        logger.info("📊 System Information:")
        logger.info(f"   CPU Cores: {cpu_count}")
        logger.info(f"   Memory: {memory_gb:.1f} GB")
        logger.info(f"   Free Disk: {disk_free_gb:.1f} GB")
        
        # Recommendations
        recommendations = get_system_recommendations()
        logger.info("💡 Performance Recommendations:")
        logger.info(f"   Recommended Workers: {recommendations['recommended_workers']}")
        logger.info(f"   Recommended Cache: {recommendations['recommended_cache_mb']} MB")
        logger.info(f"   Suggested Preset: {recommendations['preset']}")
        
        # Warnings
        if memory_gb < 4:
            logger.warning("⚠️ Low memory detected. Consider using 'memory_efficient' preset.")
        
        if disk_free_gb < 5:
            logger.warning("⚠️ Low disk space. Consider cleaning up temporary files.")
        
        if cpu_count < 4:
            logger.warning("⚠️ Low CPU count. Performance may be limited.")
        
        return True
        
    except ImportError:
        logger.warning("⚠️ psutil not available, skipping system checks")
        return True
    except Exception as e:
        logger.error(f"❌ System check failed: {e}")
        return False

def setup_environment():
    """Setup environment variables and paths"""
    try:
        # Set environment variables for optimization
        os.environ['PYTHONUNBUFFERED'] = '1'
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        
        # Set optimal Python settings
        if hasattr(sys, 'set_int_max_str_digits'):
            sys.set_int_max_str_digits(0)  # Remove string length limit
        
        # Create temp directory if it doesn't exist
        temp_dir = Path(settings.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set optimal file buffer size
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        logger.info("✅ Environment configured for optimal performance")
        
    except Exception as e:
        logger.error(f"❌ Environment setup failed: {e}")

def print_startup_banner():
    """Print startup banner with configuration info"""
    print("\n" + "="*80)
    print("🚀 FAST AI DOCUMENT PROCESSOR - ULTRA HIGH PERFORMANCE")
    print("="*80)
    print(f"Version: {settings.app_version}")
    print(f"Host: {settings.host}:{settings.port}")
    print(f"Workers: {settings.max_workers}")
    print(f"Cache: {settings.cache_max_memory_mb} MB")
    print(f"Streaming: {'✅' if settings.enable_streaming else '❌'}")
    print(f"Parallel AI: {'✅' if settings.enable_parallel_ai else '❌'}")
    print(f"UVLoop: {'✅' if settings.enable_uvloop else '❌'}")
    print(f"Redis Cache: {'✅' if settings.cache_redis_url else '❌'}")
    print("="*80)
    
    # Print performance tips
    tips = settings.get_performance_tips()
    if tips:
        print("\n💡 Performance Status:")
        for tip in tips:
            print(f"   {tip}")
    
    print("\n🌐 API Endpoints:")
    print(f"   • Health Check: http://{settings.host}:{settings.port}/health")
    print(f"   • Metrics: http://{settings.host}:{settings.port}/metrics")
    print(f"   • Process Document: http://{settings.host}:{settings.port}/process")
    print(f"   • Batch Process: http://{settings.host}:{settings.port}/process/batch")
    print(f"   • API Docs: http://{settings.host}:{settings.port}/docs")
    print("="*80 + "\n")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def main():
    """Main startup function"""
    try:
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Print banner
        print_startup_banner()
        
        # Check system requirements
        if not check_system_requirements():
            logger.error("❌ System requirements check failed")
            sys.exit(1)
        
        # Setup environment
        setup_environment()
        
        # Apply performance optimizations
        setup_performance_optimizations()
        
        # Get optimized settings
        optimized_settings = settings.get_optimized_settings()
        
        logger.info("🚀 Starting Fast AI Document Processor...")
        logger.info(f"📡 Server will be available at http://{settings.host}:{settings.port}")
        
        # Start the server with optimized settings
        uvicorn.run(
            "fast_main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            workers=1,  # Single worker for async processing
            loop="asyncio" if not settings.enable_uvloop else "uvloop",
            access_log=True,
            log_level=settings.log_level.lower(),
            # Additional optimizations
            limit_concurrency=settings.max_concurrent_files,
            limit_max_requests=1000,
            timeout_keep_alive=30,
            # Enable HTTP/2 if available
            http="h11",  # Use h11 for better performance
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

















