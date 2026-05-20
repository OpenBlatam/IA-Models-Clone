from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import sys
import asyncio
import structlog
from pathlib import Path
from core.config import get_config
from performance_monitor import monitor, start_monitoring
from cache_manager import initialize_cache
from async_processor import initialize_processor
from load_balancer import initialize_load_balancer
from cdn_manager import initialize_cdn_manager
from database.connection import initialize_database, close_database
from api_refactored import app
        from cache_manager import cleanup_cache
        from async_processor import cleanup_processor
        from load_balancer import cleanup_load_balancer
        from cdn_manager import cleanup_cdn_manager
        import uvicorn
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Startup script for OS Content UGC Video Generator
Initializes performance monitoring, cache, async processor, load balancer, CDN, and database
"""


# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


# Configure structured logging
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

logger = structlog.get_logger("os_content.startup")

async def initialize_system():
    """Initialize all system components"""
    try:
        config = get_config()
        
        # Create necessary directories
        Path("logs").mkdir(exist_ok=True)
        Path("cache").mkdir(exist_ok=True)
        Path("cdn_cache").mkdir(exist_ok=True)
        Path("ssl").mkdir(exist_ok=True)
        Path("uploads").mkdir(exist_ok=True)
        
        # Initialize performance monitoring
        if config.monitoring.enabled:
            await start_monitoring()
            logger.info("Performance monitoring initialized")
        
        # Initialize cache
        await initialize_cache()
        logger.info("Cache initialized")
        
        # Initialize async processor
        max_concurrent = config.processor.max_concurrent
        await initialize_processor(max_concurrent)
        logger.info("Async processor initialized")
        
        # Initialize load balancer if backend servers configured
        backend_servers = config.load_balancer.backend_servers
        if backend_servers:
            await initialize_load_balancer(backend_servers, algorithm=config.load_balancer.algorithm)
            logger.info("Load balancer initialized")
        
        # Initialize CDN manager
        if config.cdn.cdn_url:
            await initialize_cdn_manager(cdn_url=config.cdn.cdn_url)
            logger.info("CDN manager initialized")
        
        # Initialize database
        await initialize_database()
        logger.info("Database initialized")
        
        logger.info("All system components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

async def cleanup_system():
    """Cleanup all system components"""
    try:
        
        await cleanup_cache()
        await cleanup_processor()
        await cleanup_load_balancer()
        await cleanup_cdn_manager()
        await close_database()
        
        logger.info("System cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def main():
    """Start the server with all optimizations and scaling"""
    try:
        # Run initialization
        asyncio.run(initialize_system())
        
        # Get configuration
        config = get_config()
        
        # Start server
        uvicorn.run(
            "api_refactored:app",
            host=config.server.host,
            port=config.server.port,
            workers=config.server.workers,
            log_level=config.server.log_level,
            access_log=True,
            reload=config.server.debug
        )
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup on shutdown
        asyncio.run(cleanup_system())

match __name__:
    case "__main__":
    main() 