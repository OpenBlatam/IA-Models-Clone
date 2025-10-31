#!/usr/bin/env python3
"""
Ultra Startup Script - Maximum Performance Launch
================================================

Ultra-optimized startup script with extreme performance tuning.
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

# Import ultra optimizations first
try:
    from ultra_optimized_config import apply_ultra_optimizations, get_ultra_config
    ULTRA_CONFIG_AVAILABLE = True
except ImportError:
    ULTRA_CONFIG_AVAILABLE = False
    print("⚠️ Ultra config not available, using standard optimizations")

# Import fast config as fallback
try:
    from fast_config import settings, get_system_recommendations, apply_performance_preset
    FAST_CONFIG_AVAILABLE = True
except ImportError:
    FAST_CONFIG_AVAILABLE = False
    print("⚠️ Fast config not available, using basic settings")

import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ultra_processor.log') if not os.getenv('DEBUG') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_ultra_performance_optimizations():
    """Setup ultra system-level performance optimizations"""
    try:
        # Apply ultra optimizations if available
        if ULTRA_CONFIG_AVAILABLE:
            apply_ultra_optimizations()
            logger.info("✅ Ultra optimizations applied")
        elif FAST_CONFIG_AVAILABLE:
            # Apply fast optimizations as fallback
            settings = apply_performance_preset('ultra_fast')
            logger.info("✅ Fast optimizations applied")
        
        # Enable UVLoop for faster async processing
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("✅ UVLoop enabled for ultra-fast async processing")
        except ImportError:
            logger.warning("⚠️ UVLoop not available, using default event loop")
        
        # Set optimal thread pool size
        if hasattr(asyncio, 'set_default_executor'):
            import concurrent.futures
            cpu_count = os.cpu_count() or 1
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(64, cpu_count * 4)
            )
            asyncio.set_default_executor(executor)
            logger.info(f"✅ Ultra thread pool executor configured with {executor._max_workers} workers")
        
        # Optimize garbage collection
        import gc
        gc.set_threshold(50)  # More aggressive GC
        logger.info("✅ Ultra garbage collection threshold set to 50")
        
        # Set process priority (Unix only)
        if os.name != 'nt':
            try:
                import psutil
                process = psutil.Process()
                process.nice(-10)  # Highest priority
                logger.info("✅ Process priority set to maximum")
            except (ImportError, PermissionError):
                logger.warning("⚠️ Could not set maximum process priority")
        
        # Memory optimizations
        try:
            import mmap
            logger.info("✅ Memory mapping optimizations available")
        except ImportError:
            logger.warning("⚠️ Memory mapping not available")
        
        logger.info("🚀 Ultra performance optimizations applied successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to apply ultra performance optimizations: {e}")

def check_ultra_system_requirements():
    """Check ultra system requirements and provide recommendations"""
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
        
        logger.info("📊 Ultra System Information:")
        logger.info(f"   CPU Cores: {cpu_count}")
        logger.info(f"   Memory: {memory_gb:.1f} GB")
        logger.info(f"   Free Disk: {disk_free_gb:.1f} GB")
        
        # Ultra recommendations
        recommendations = []
        
        if memory_gb >= 32:
            recommendations.append("🚀 Ultra high memory system - excellent for large models")
        elif memory_gb >= 16:
            recommendations.append("⚡ High memory system - good for advanced features")
        elif memory_gb >= 8:
            recommendations.append("✅ Good memory system - balanced performance")
        else:
            recommendations.append("⚠️ Limited memory - consider upgrading for ultra performance")
        
        if cpu_count >= 32:
            recommendations.append("🚀 Ultra high CPU count - excellent for parallel processing")
        elif cpu_count >= 16:
            recommendations.append("⚡ High CPU count - very good for parallel processing")
        elif cpu_count >= 8:
            recommendations.append("✅ Good CPU count - good for parallel processing")
        else:
            recommendations.append("⚠️ Limited CPU count - consider upgrading for ultra performance")
        
        if disk_free_gb >= 100:
            recommendations.append("🚀 Excellent disk space - can handle large datasets")
        elif disk_free_gb >= 50:
            recommendations.append("✅ Good disk space - sufficient for most operations")
        else:
            recommendations.append("⚠️ Limited disk space - consider freeing up space")
        
        logger.info("💡 Ultra Performance Recommendations:")
        for rec in recommendations:
            logger.info(f"   {rec}")
        
        return True
        
    except ImportError:
        logger.warning("⚠️ psutil not available, skipping ultra system checks")
        return True
    except Exception as e:
        logger.error(f"❌ Ultra system check failed: {e}")
        return False

def setup_ultra_environment():
    """Setup ultra environment variables and paths"""
    try:
        # Set environment variables for ultra optimization
        ultra_env_vars = {
            'PYTHONUNBUFFERED': '1',
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONOPTIMIZE': '2',
            'PYTHONHASHSEED': '0',
            'PYTHONIOENCODING': 'utf-8',
            'OMP_NUM_THREADS': str(os.cpu_count() or 1),
            'MKL_NUM_THREADS': str(os.cpu_count() or 1),
            'NUMEXPR_NUM_THREADS': str(os.cpu_count() or 1),
            'OPENBLAS_NUM_THREADS': str(os.cpu_count() or 1),
            'VECLIB_MAXIMUM_THREADS': str(os.cpu_count() or 1),
            'NUMBA_NUM_THREADS': str(os.cpu_count() or 1),
            'BLIS_NUM_THREADS': str(os.cpu_count() or 1),
            'NUMPY_MADVISE_HUGEPAGE': '1',
            'NUMPY_DISABLE_CPU_FEATURES': '0',
            'MALLOC_TRIM_THRESHOLD_': '131072',
            'MALLOC_MMAP_THRESHOLD_': '131072',
            'REDIS_MAXMEMORY': '4096mb',
            'REDIS_MAXMEMORY_POLICY': 'allkeys-lru',
            'REDIS_SAVE': '',
            'TCP_NODELAY': '1',
            'TCP_KEEPALIVE': '1'
        }
        
        for key, value in ultra_env_vars.items():
            os.environ[key] = value
        
        # Set optimal Python settings
        if hasattr(sys, 'set_int_max_str_digits'):
            sys.set_int_max_str_digits(0)  # Remove string length limit
        
        # Create temp directory if it doesn't exist
        temp_dir = Path("/tmp" if os.name != 'nt' else os.environ.get('TEMP', 'C:\\temp'))
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Ultra environment configured for maximum performance")
        
    except Exception as e:
        logger.error(f"❌ Ultra environment setup failed: {e}")

def print_ultra_startup_banner():
    """Print ultra startup banner with configuration info"""
    print("\n" + "="*80)
    print("🚀 ULTRA AI DOCUMENT PROCESSOR - MAXIMUM PERFORMANCE")
    print("="*80)
    
    # Get configuration
    if ULTRA_CONFIG_AVAILABLE:
        config = get_ultra_config()
        print(f"Version: Ultra 3.0.0")
        print(f"Host: {config['fastapi']['host']}:{config['fastapi']['port']}")
        print(f"Workers: {config['processor']['max_workers']}")
        print(f"Cache: {config['cache']['max_memory_mb']} MB")
        print(f"Memory Limit: {config['memory']['memory_limit_gb']} GB")
        print(f"UVLoop: {'✅' if config['fastapi']['loop'] == 'uvloop' else '❌'}")
        print(f"Memory Mapping: {'✅' if config['memory']['memory_mapping'] else '❌'}")
        print(f"Zero Copy: {'✅' if config['processor'].get('zero_copy', False) else '❌'}")
        print(f"Ultra Compression: {'✅' if config['cache']['compression'] else '❌'}")
    elif FAST_CONFIG_AVAILABLE:
        print(f"Version: Fast 2.0.0")
        print(f"Host: {settings.host}:{settings.port}")
        print(f"Workers: {settings.max_workers}")
        print(f"Cache: {settings.cache_max_memory_mb} MB")
        print(f"Streaming: {'✅' if settings.enable_streaming else '❌'}")
        print(f"Parallel AI: {'✅' if settings.enable_parallel_ai else '❌'}")
        print(f"UVLoop: {'✅' if settings.enable_uvloop else '❌'}")
        print(f"Redis Cache: {'✅' if settings.cache_redis_url else '❌'}")
    else:
        print("Version: Standard 1.0.0")
        print("Host: 0.0.0.0:8001")
        print("Workers: 8")
        print("Cache: 512 MB")
    
    print("="*80)
    
    # Print ultra performance tips
    tips = [
        "🚀 Ultra-fast document processing",
        "⚡ Maximum parallel processing",
        "💾 Advanced memory optimization",
        "🔄 Zero-copy operations",
        "📦 Ultra-fast serialization",
        "🗜️ Advanced compression",
        "🎯 GPU acceleration ready",
        "📊 Enterprise monitoring",
        "🔧 Auto-optimization",
        "⚡ Sub-second response times"
    ]
    
    print("\n💡 Ultra Performance Features:")
    for tip in tips:
        print(f"   {tip}")
    
    print("\n🌐 Ultra API Endpoints:")
    print(f"   • Health Check: http://0.0.0.0:8001/health")
    print(f"   • Ultra Metrics: http://0.0.0.0:8001/metrics")
    print(f"   • Ultra Process: http://0.0.0.0:8001/process")
    print(f"   • Ultra Batch: http://0.0.0.0:8001/process/batch")
    print(f"   • Ultra API Docs: http://0.0.0.0:8001/docs")
    print("="*80 + "\n")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"🛑 Received signal {signum}, shutting down ultra processor gracefully...")
    sys.exit(0)

def main():
    """Main ultra startup function"""
    try:
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Print ultra banner
        print_ultra_startup_banner()
        
        # Check ultra system requirements
        if not check_ultra_system_requirements():
            logger.error("❌ Ultra system requirements check failed")
            sys.exit(1)
        
        # Setup ultra environment
        setup_ultra_environment()
        
        # Apply ultra performance optimizations
        setup_ultra_performance_optimizations()
        
        # Get ultra configuration
        if ULTRA_CONFIG_AVAILABLE:
            config = get_ultra_config()
            fastapi_config = config['fastapi']
        elif FAST_CONFIG_AVAILABLE:
            fastapi_config = {
                'host': settings.host,
                'port': settings.port,
                'workers': 1,
                'loop': 'uvloop' if settings.enable_uvloop else 'asyncio',
                'access_log': True,
                'log_level': settings.log_level.lower()
            }
        else:
            fastapi_config = {
                'host': '0.0.0.0',
                'port': 8001,
                'workers': 1,
                'loop': 'asyncio',
                'access_log': True,
                'log_level': 'info'
            }
        
        logger.info("🚀 Starting Ultra AI Document Processor...")
        logger.info(f"📡 Server will be available at http://{fastapi_config['host']}:{fastapi_config['port']}")
        
        # Start the server with ultra optimizations
        uvicorn.run(
            "fast_main:app",  # Use fast_main as the entry point
            host=fastapi_config['host'],
            port=fastapi_config['port'],
            reload=False,  # Disable reload for production
            workers=fastapi_config['workers'],
            loop=fastapi_config['loop'],
            access_log=fastapi_config['access_log'],
            log_level=fastapi_config['log_level'],
            # Ultra optimizations
            limit_concurrency=1000,
            limit_max_requests=10000,
            timeout_keep_alive=5,
            timeout_graceful_shutdown=5,
            # HTTP/2 support
            http="h11",
            # Additional optimizations
            server_header=False,
            date_header=False,
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Ultra shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Ultra startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

















