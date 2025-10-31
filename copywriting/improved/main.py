"""
Main Entry Point
===============

Production-ready entry point for the copywriting service.
"""

import sys
import logging
from pathlib import Path

import uvicorn
from .app import app
from .config import get_api_settings, get_logging_settings

# Configure logging
logging_settings = get_logging_settings()
logging.basicConfig(
    level=getattr(logging, logging_settings.level),
    format=logging_settings.format,
    handlers=[
        logging.StreamHandler(),
        *([logging.FileHandler(logging_settings.file_path)] if logging_settings.file_path else [])
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main application entry point"""
    try:
        api_settings = get_api_settings()
        
        logger.info(f"Starting Copywriting Service on {api_settings.host}:{api_settings.port}")
        logger.info(f"Environment: {api_settings.environment}")
        logger.info(f"Workers: {api_settings.workers}")
        logger.info(f"Debug mode: {api_settings.debug}")
        
        uvicorn.run(
            app,
            host=api_settings.host,
            port=api_settings.port,
            workers=api_settings.workers,
            reload=api_settings.reload,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()






























