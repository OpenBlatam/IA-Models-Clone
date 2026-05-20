"""
Gamma App - Main API Application
FastAPI application for AI-powered content generation
"""

import logging
import sys
import uvicorn
from typing import Optional

from .app_factory import create_app
from ..utils.config import get_settings, get_log_config
from ..utils.logging_config import setup_logging

_logger: Optional[logging.Logger] = None
_app = None


def _configure_logging() -> logging.Logger:
    """Configure application logging and return logger instance"""
    global _logger
    
    if _logger is not None:
        return _logger
    
    try:
        log_config = get_log_config()
        setup_logging({
            "log_dir": "logs",
            "console_level": log_config["level"],
            "file_level": log_config["level"],
            "app_level": log_config["level"],
            "api_level": log_config["level"],
            "root_level": "WARNING"
        })
    except Exception as e:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.error(
            "Failed to setup advanced logging",
            extra={"error": str(e)},
            exc_info=True
        )
    
    _logger = logging.getLogger(__name__)
    return _logger


def _create_application():
    """Create and return the FastAPI application instance"""
    global _app
    
    if _app is not None:
        return _app
    
    logger = _configure_logging()
    
    try:
        _app = create_app()
        logger.info("Application created successfully")
        return _app
    except Exception as e:
        logger.error(
            "Failed to create application",
            extra={"error": str(e)},
            exc_info=True
        )
        sys.exit(1)


def _run_server() -> None:
    """Run the uvicorn server"""
    logger = _configure_logging()
    app_instance = _create_application()
    settings = get_settings()
    
    logger.info(
        "Starting Gamma App API server",
        extra={
            "host": settings.api_host,
            "port": settings.api_port,
            "debug": settings.debug,
            "environment": settings.environment,
            "version": settings.app_version
        }
    )
    
    uvicorn_config = {
        "host": settings.api_host,
        "port": settings.api_port,
        "log_level": settings.log_level.lower(),
        "access_log": True
    }
    
    try:
        if settings.debug:
            uvicorn.run(
                "gamma_app.api.main:app",
                reload=True,
                **uvicorn_config
            )
        else:
            uvicorn.run(
                app_instance,
                **uvicorn_config
            )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(
            "Server error",
            extra={"error": str(e)},
            exc_info=True
        )
        sys.exit(1)


_configure_logging()
app = _create_application()


if __name__ == "__main__":
    _run_server()
