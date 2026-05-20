"""
Application lifecycle management

Handles startup and shutdown procedures for the FastAPI application.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI

from ..config import settings
from .database import (
    init_database,
    verify_database_connection,
    DatabaseManager
)
from ..utils.logging_config import StructuredLogger

logger = StructuredLogger(__name__)

# Global database manager instance
_db_manager: DatabaseManager = DatabaseManager()


def startup_handler() -> None:
    """
    Handle application startup procedures.
    
    Performs:
    - Database initialization
    - Connection verification
    - Performance optimizations
    - Logging of startup information
    
    Raises:
        Exception: If critical startup operations fail
    """
    logger.info(
        "Starting application",
        app_name=settings.app_name,
        version=settings.app_version,
        debug=settings.debug
    )
    
    try:
        # Initialize database
        _db_manager.initialize()
        logger.info("Database initialized successfully")
        
        # Verify connection
        if _db_manager.verify_connection():
            logger.info("Database connection verified")
        else:
            logger.warning(
                "Database connection verification failed",
                database_url=settings.database_url.split("@")[-1] if "@" in settings.database_url else "local"
            )
            # Don't raise - allow app to start but log warning
        
        # Configure performance optimizations (non-critical)
        try:
            from .performance_config import configure_pytorch_performance
            configure_pytorch_performance()
            logger.debug("PyTorch performance configured")
            
            # Additional CUDA optimizations
            try:
                from ...utils import optimize_cuda_settings, enable_tensor_cores
                optimize_cuda_settings()
                enable_tensor_cores()
                logger.debug("CUDA optimizations enabled")
            except ImportError:
                logger.debug("CUDA utilities not available")
            except Exception as e:
                logger.warning("Failed to configure CUDA optimizations", error=str(e))
        except ImportError:
            logger.debug("PyTorch not available, skipping performance config")
        except Exception as e:
            logger.warning("Failed to configure performance", error=str(e), error_type=type(e).__name__)
        
        logger.info("Application startup completed successfully")
            
    except Exception as e:
        logger.exception(
            "Critical error during startup",
            error=str(e),
            error_type=type(e).__name__
        )
        raise


def shutdown_handler() -> None:
    """
    Handle application shutdown procedures.
    
    Performs cleanup operations:
    - Close database connections
    - Clean up resources
    - Log shutdown information
    """
    logger.info(
        "Shutting down application",
        app_name=settings.app_name,
        version=settings.app_version
    )
    
    try:
        # Close database connections
        try:
            _db_manager.close_all()
            logger.debug("Database connections closed")
        except Exception as e:
            logger.warning("Error closing database connections", error=str(e))
        
        # Add any additional cleanup operations here if needed
        
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(
            "Error during shutdown",
            error=str(e),
            error_type=type(e).__name__
        )
        # Don't raise - allow shutdown to complete


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application.
    
    Handles:
    - Startup procedures
    - Resource initialization
    - Shutdown cleanup
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None: Control is yielded to the application
    """
    # Startup
    startup_handler()
    
    yield
    
    # Shutdown
    shutdown_handler()


def get_database_manager() -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Returns:
        DatabaseManager: Global database manager instance
    """
    return _db_manager

