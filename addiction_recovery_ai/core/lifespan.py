"""
Lifespan context manager for FastAPI application
Replaces @app.on_event("startup") and @app.on_event("shutdown")
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Any
import logging

from config.app_config import get_config

logger = logging.getLogger(__name__)

_db_pool: Optional[Any] = None
_cache_client: Optional[Any] = None


@asynccontextmanager
async def lifespan(app) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for application startup and shutdown
    
    This replaces the deprecated @app.on_event("startup") and @app.on_event("shutdown")
    """
    # Startup
    logger.info("Starting Addiction Recovery AI service...")
    
    try:
        # Initialize services, connections, etc.
        await initialize_services()
        await initialize_database()
        await initialize_cache()
        
        logger.info("Service started successfully")
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down Addiction Recovery AI service...")
        
        await cleanup_cache()
        await cleanup_database()
        await cleanup_services()
        
        logger.info("Service shut down successfully")


async def initialize_services() -> None:
    """Initialize application services"""
    logger.info("Initializing services...")
    
    try:
        from performance.warmup import initialize_warmup
        await initialize_warmup()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Warmup failed: {str(e)}")
    
    try:
        from modules.module_registry import get_module_registry
        registry = get_module_registry()
        await registry.initialize_all()
    except ImportError:
        logger.debug("Module registry not available")
    except Exception as e:
        logger.warning(f"Module registry initialization failed: {str(e)}")


async def initialize_database() -> None:
    """Initialize database connections"""
    global _db_pool
    logger.info("Initializing database...")
    
    config = get_config()
    if not config.database_url:
        logger.info("No database URL configured, skipping database initialization")
        return
    
    try:
        if config.database_url.startswith("postgresql") or config.database_url.startswith("postgres"):
            import asyncpg
            _db_pool = await asyncpg.create_pool(
                config.database_url,
                min_size=1,
                max_size=config.database_pool_size,
                command_timeout=60
            )
            logger.info("PostgreSQL connection pool created")
        elif config.database_url.startswith("sqlite"):
            import aiosqlite
            _db_pool = await aiosqlite.connect(config.database_url.replace("sqlite:///", ""))
            logger.info("SQLite connection created")
        else:
            logger.warning(f"Unsupported database URL: {config.database_url}")
    except ImportError as e:
        logger.warning(f"Database driver not available: {str(e)}")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")


async def initialize_cache() -> None:
    """Initialize cache"""
    global _cache_client
    logger.info("Initializing cache...")
    
    config = get_config()
    if not config.cache_enabled:
        logger.info("Cache disabled, skipping cache initialization")
        return
    
    try:
        from infrastructure.cache import RedisCacheService, InMemoryCacheService
        from config.aws_settings import get_aws_settings
        
        aws_settings = get_aws_settings()
        if aws_settings.redis_endpoint or config.database_url and "redis" in config.database_url.lower():
            _cache_client = RedisCacheService()
            logger.info("Redis cache initialized")
        else:
            _cache_client = InMemoryCacheService()
            logger.info("In-memory cache initialized")
    except ImportError:
        try:
            from infrastructure.cache import InMemoryCacheService
            _cache_client = InMemoryCacheService()
            logger.info("In-memory cache initialized (Redis not available)")
        except ImportError:
            logger.warning("Cache services not available")
    except Exception as e:
        logger.warning(f"Cache initialization failed: {str(e)}")


async def cleanup_services() -> None:
    """Cleanup application services"""
    logger.info("Cleaning up services...")
    
    try:
        from modules.module_registry import get_module_registry
        registry = get_module_registry()
        await registry.shutdown_all()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Module registry shutdown failed: {str(e)}")


async def cleanup_database() -> None:
    """Cleanup database connections"""
    global _db_pool
    logger.info("Cleaning up database...")
    
    if _db_pool:
        try:
            if hasattr(_db_pool, "close"):
                await _db_pool.close()
                logger.info("Database connection pool closed")
            elif hasattr(_db_pool, "close"):
                await _db_pool.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")
        finally:
            _db_pool = None


async def cleanup_cache() -> None:
    """Cleanup cache"""
    global _cache_client
    logger.info("Cleaning up cache...")
    
    if _cache_client:
        try:
            if hasattr(_cache_client, "client") and hasattr(_cache_client.client, "close"):
                _cache_client.client.close()
                logger.info("Cache connection closed")
            elif hasattr(_cache_client, "close"):
                await _cache_client.close()
                logger.info("Cache connection closed")
        except Exception as e:
            logger.warning(f"Error closing cache connection: {str(e)}")
        finally:
            _cache_client = None

