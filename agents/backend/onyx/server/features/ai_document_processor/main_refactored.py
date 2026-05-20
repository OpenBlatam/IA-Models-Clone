#!/usr/bin/env python3
"""
Main Refactored Application - Modern Architecture
===============================================

Main entry point for the refactored AI Document Processor with clean architecture.
"""

import asyncio
import logging
import sys
import signal
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import get_config_manager
from src.core.exceptions import ConfigurationError
from src.api.app import create_app, run_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ai_processor.log')
    ]
)
logger = logging.getLogger(__name__)


def print_startup_banner():
    """Print startup banner."""
    print("\n" + "="*80)
    print("🚀 AI DOCUMENT PROCESSOR - REFACTORED ARCHITECTURE")
    print("="*80)
    print("Modern, clean architecture with separation of concerns")
    print("Version: 3.0.0")
    print("="*80 + "\n")


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def validate_environment():
    """Validate environment and dependencies."""
    logger.info("🔍 Validating environment...")
    
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            raise ConfigurationError("Python 3.8+ required")
        
        # Check required packages
        required_packages = [
            'fastapi',
            'uvicorn',
            'pydantic',
            'aiofiles'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                raise ConfigurationError(f"Required package not found: {package}")
        
        logger.info("✅ Environment validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment validation failed: {e}")
        return False


def setup_application():
    """Setup and configure the application."""
    logger.info("⚙️ Setting up application...")
    
    try:
        # Get configuration manager
        config_manager = get_config_manager()
        logger.info("✅ Configuration manager initialized")
        
        # Validate configuration
        config_manager._validate_configs()
        logger.info("✅ Configuration validated")
        
        # Create FastAPI app
        app = create_app(config_manager)
        logger.info("✅ FastAPI application created")
        
        return app, config_manager
        
    except Exception as e:
        logger.error(f"❌ Application setup failed: {e}")
        raise


def main():
    """Main application entry point."""
    try:
        # Print startup banner
        print_startup_banner()
        
        # Setup signal handlers
        setup_signal_handlers()
        
        # Validate environment
        if not validate_environment():
            sys.exit(1)
        
        # Setup application
        app, config_manager = setup_application()
        
        # Get server configuration
        server_config = config_manager.get('server')
        
        logger.info("🚀 Starting AI Document Processor API...")
        logger.info(f"📡 Server will be available at http://{server_config.host}:{server_config.port}")
        logger.info(f"📚 API documentation at http://{server_config.host}:{server_config.port}/docs")
        logger.info(f"❤️ Health check at http://{server_config.host}:{server_config.port}/api/v1/health")
        
        # Run the application
        run_app(config_manager)
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except ConfigurationError as e:
        logger.error(f"❌ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



















