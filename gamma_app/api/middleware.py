"""
Middleware Configuration
Configures CORS, security, and other middleware for the FastAPI application
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from ..utils.config import get_settings, get_cors_config, is_production

logger = logging.getLogger(__name__)

def setup_middleware(app: FastAPI) -> None:
    """Configure middleware for the application"""
    settings = get_settings()
    cors_config = get_cors_config()
    
    try:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config["origins"] if is_production() else ["*"],
            allow_credentials=cors_config["allow_credentials"],
            allow_methods=cors_config["allow_methods"],
            allow_headers=cors_config["allow_headers"],
        )
        logger.info("CORS middleware configured")
        
        if is_production():
            allowed_hosts = (
                settings.cors_origins 
                if isinstance(settings.cors_origins, list) 
                else ["*"]
            )
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=allowed_hosts
            )
            logger.info("TrustedHost middleware configured for production")
    except Exception as e:
        logger.error(f"Failed to setup middleware: {e}", exc_info=True)
        raise
