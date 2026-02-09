"""Application lifespan management."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None: Control flow for application lifecycle
    """
    # Startup
    await startup()
    
    yield
    
    # Shutdown
    await shutdown()


async def startup() -> None:
    """Perform startup tasks."""
    logger.info("Starting Plastic Surgery Visualization AI service...")
    
    # Ensure storage directories exist
    await ensure_directories()
    
    # Check dependencies
    check_dependencies()
    
    logger.info("Service startup complete")


async def shutdown() -> None:
    """Perform shutdown tasks."""
    logger.info("Shutting down Plastic Surgery Visualization AI service...")
    logger.info("Shutdown complete")


async def ensure_directories() -> None:
    """Ensure required directories exist."""
    upload_dir = Path(settings.upload_dir)
    output_dir = Path(settings.output_dir)
    cache_dir = Path("./storage/cache")
    
    for directory in [upload_dir, output_dir, cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def check_dependencies() -> None:
    """Check if required dependencies are available."""
    try:
        import PIL
        import numpy
        logger.info("Core dependencies available: PIL, numpy")
    except ImportError as e:
        logger.warning(f"Missing dependency: {e}")
    
    try:
        import cv2
        logger.info("OpenCV available for advanced image processing")
    except ImportError:
        logger.info("OpenCV not available (optional)")

