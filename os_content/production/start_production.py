from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import os
import sys
import asyncio
import signal
import structlog
from pathlib import Path
from contextlib import asynccontextmanager
from production.config import get_production_config
from production.deployment import deployment_manager
from production.monitoring import production_monitor
from production.security import security_manager
from database.connection import initialize_database, close_database
from cache_manager import initialize_cache, cleanup_cache
from async_processor import initialize_processor, cleanup_processor
from load_balancer import initialize_load_balancer, cleanup_load_balancer
from cdn_manager import initialize_cdn_manager, cleanup_cdn_manager
from api_refactored import app
                from fastapi import HTTPException, status
            import uvicorn
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Production Startup Script for OS Content UGC Video Generator
Initializes all production components and starts the application
"""


# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure structured logging for production
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

logger = structlog.get_logger("os_content.production")

class ProductionApplication:
    """Production application manager"""
    
    def __init__(self) -> Any:
        self.config = get_production_config()
        self.shutdown_event = asyncio.Event()
        self.cleanup_tasks = []
    
    async def initialize(self) -> Any:
        """Initialize all production components"""
        try:
            logger.info("Initializing production application")
            
            # Create necessary directories
            await self._create_directories()
            
            # Initialize database
            await initialize_database()
            logger.info("Database initialized")
            
            # Initialize cache
            await initialize_cache()
            logger.info("Cache initialized")
            
            # Initialize async processor
            await initialize_processor(self.config.max_concurrent_tasks)
            logger.info("Async processor initialized")
            
            # Initialize load balancer if configured
            if hasattr(self.config, 'load_balancer') and self.config.load_balancer.backend_servers:
                await initialize_load_balancer(
                    self.config.load_balancer.backend_servers,
                    algorithm=self.config.load_balancer.algorithm
                )
                logger.info("Load balancer initialized")
            
            # Initialize CDN manager
            if self.config.cdn_url:
                await initialize_cdn_manager(cdn_url=self.config.cdn_url)
                logger.info("CDN manager initialized")
            
            # Start production monitoring
            await production_monitor.start_monitoring()
            logger.info("Production monitoring started")
            
            # Setup security
            await self._setup_security()
            logger.info("Security setup completed")
            
            # Setup periodic cleanup tasks
            await self._setup_cleanup_tasks()
            logger.info("Cleanup tasks configured")
            
            logger.info("Production application initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize production application: {e}")
            await self.cleanup()
            raise
    
    async def _create_directories(self) -> Any:
        """Create necessary directories"""
        directories = [
            self.config.upload_dir,
            Path(self.config.log_file).parent,
            self.config.backup_path,
            "/var/cache/os_content",
            "/var/tmp/os_content",
            "/var/lib/os_content",
            "/var/log/os_content"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    async def _setup_security(self) -> Any:
        """Setup security components"""
        # Add security headers middleware
        @app.middleware("http")
        async def security_middleware(request, call_next) -> Any:
            response = await call_next(request)
            
            # Add security headers
            for header, value in security_manager.get_security_headers().items():
                response.headers[header] = value
            
            # Add CORS headers
            origin = request.headers.get("origin")
            if origin in security_manager.cors_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
            
            return response
        
        # Add rate limiting middleware
        @app.middleware("http")
        async def rate_limit_middleware(request, call_next) -> Any:
            client_ip = request.client.host if request.client else "unknown"
            
            if not security_manager.check_rate_limit(client_ip):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            response = await call_next(request)
            
            # Add rate limit headers
            rate_limit_info = security_manager.get_rate_limit_info(client_ip)
            response.headers["X-RateLimit-Remaining"] = str(rate_limit_info["remaining"])
            response.headers["X-RateLimit-Reset"] = rate_limit_info["reset_time"].isoformat()
            
            return response
    
    async def _setup_cleanup_tasks(self) -> Any:
        """Setup periodic cleanup tasks"""
        # Security cleanup task
        async def security_cleanup():
            
    """security_cleanup function."""
while not self.shutdown_event.is_set():
                try:
                    security_manager.cleanup_expired_tokens()
                    await asyncio.sleep(300)  # Every 5 minutes
                except Exception as e:
                    logger.error(f"Security cleanup error: {e}")
                    await asyncio.sleep(60)
        
        # Database cleanup task
        async def database_cleanup():
            
    """database_cleanup function."""
while not self.shutdown_event.is_set():
                try:
                    # Cleanup old records, logs, etc.
                    await asyncio.sleep(3600)  # Every hour
                except Exception as e:
                    logger.error(f"Database cleanup error: {e}")
                    await asyncio.sleep(300)
        
        # Cache cleanup task
        async def cache_cleanup():
            
    """cache_cleanup function."""
while not self.shutdown_event.is_set():
                try:
                    # Cache cleanup is handled by cache manager
                    await asyncio.sleep(1800)  # Every 30 minutes
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    await asyncio.sleep(300)
        
        # Start cleanup tasks
        self.cleanup_tasks.extend([
            asyncio.create_task(security_cleanup()),
            asyncio.create_task(database_cleanup()),
            asyncio.create_task(cache_cleanup())
        ])
    
    async def start(self) -> Any:
        """Start the production application"""
        try:
            logger.info("Starting production application")
            
            # Initialize components
            await self.initialize()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Start uvicorn server
            config = uvicorn.Config(
                app=app,
                host=self.config.host,
                port=self.config.port,
                workers=self.config.workers,
                worker_class=self.config.worker_class,
                log_level=self.config.log_level,
                access_log=True,
                reload=False,  # Disable reload in production
                ssl_keyfile=self.config.ssl_key_path if self.config.ssl_key_path else None,
                ssl_certfile=self.config.ssl_cert_path if self.config.ssl_cert_path else None,
                proxy_headers=True,
                forwarded_allow_ips="*"
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start production application: {e}")
            await self.cleanup()
            raise
    
    def _setup_signal_handlers(self) -> Any:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame) -> Any:
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def cleanup(self) -> Any:
        """Cleanup all resources"""
        try:
            logger.info("Starting production application cleanup")
            
            # Set shutdown event
            self.shutdown_event.set()
            
            # Cancel cleanup tasks
            for task in self.cleanup_tasks:
                task.cancel()
            
            # Wait for cleanup tasks to complete
            if self.cleanup_tasks:
                await asyncio.gather(*self.cleanup_tasks, return_exceptions=True)
            
            # Stop monitoring
            await production_monitor.stop_monitoring()
            logger.info("Production monitoring stopped")
            
            # Cleanup components
            await cleanup_cache()
            await cleanup_processor()
            await cleanup_load_balancer()
            await cleanup_cdn_manager()
            await close_database()
            
            logger.info("Production application cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

@asynccontextmanager
async def production_lifespan():
    """Production application lifespan manager"""
    app_instance = ProductionApplication()
    
    try:
        await app_instance.initialize()
        yield app_instance
    finally:
        await app_instance.cleanup()

async def main():
    """Main production entry point"""
    try:
        # Validate production configuration
        config = get_production_config()
        logger.info("Production configuration validated")
        
        # Create and start production application
        app_instance = ProductionApplication()
        await app_instance.start()
        
    except KeyboardInterrupt:
        logger.info("Production application stopped by user")
    except Exception as e:
        logger.error(f"Production application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set production environment
    os.environ["ENVIRONMENT"] = "production"
    
    # Run production application
    asyncio.run(main()) 