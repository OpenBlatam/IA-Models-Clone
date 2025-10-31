"""
PDF Variantes API - Main Application Entry Point
"""

import asyncio
import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api.main import app
from utils.config import get_settings
from utils.logging_config import setup_logging

def main():
    """Main application entry point"""
    
    # Load settings
    settings = get_settings()
    
    # Setup logging
    setup_logging(
        log_level=settings.LOG_LEVEL,
        log_file=settings.LOG_FILE
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Configure uvicorn
    uvicorn_config = {
        "app": "main:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": settings.DEBUG,
        "log_level": settings.LOG_LEVEL.lower(),
        "access_log": True,
        "use_colors": True
    }
    
    # Add SSL configuration for production
    if settings.ENVIRONMENT == "production":
        uvicorn_config.update({
            "ssl_keyfile": "ssl/private.key",
            "ssl_certfile": "ssl/certificate.crt"
        })
    
    # Start server
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()