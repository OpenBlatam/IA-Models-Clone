#!/usr/bin/env python3
"""
Production Startup Script
==========================

Script to start the Bulk TruthGPT system in production mode.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config.production import ProductionConfig
from utils.logger import setup_logger

logger = setup_logger(__name__)

async def check_dependencies():
    """Check if all required dependencies are available."""
    try:
        logger.info("Checking dependencies...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")
        
        # Check required packages
        required_packages = [
            'fastapi', 'uvicorn', 'sqlalchemy', 'redis', 'torch', 'transformers'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                raise ImportError(f"Required package {package} is not installed")
        
        logger.info("All dependencies are available")
        return True
        
    except Exception as e:
        logger.error(f"Dependency check failed: {str(e)}")
        return False

async def setup_environment():
    """Setup production environment."""
    try:
        logger.info("Setting up production environment...")
        
        # Validate configuration
        if not ProductionConfig.validate_config():
            raise RuntimeError("Configuration validation failed")
        
        # Create necessary directories
        directories = [
            ProductionConfig.STORAGE_PATH,
            ProductionConfig.TEMPLATES_PATH,
            ProductionConfig.MODELS_PATH,
            ProductionConfig.KNOWLEDGE_PATH,
            "./logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Set environment variables
        os.environ['ENVIRONMENT'] = 'production'
        os.environ['PYTHONPATH'] = str(Path(__file__).parent)
        
        logger.info("Production environment setup complete")
        return True
        
    except Exception as e:
        logger.error(f"Environment setup failed: {str(e)}")
        return False

async def start_application():
    """Start the Bulk TruthGPT application."""
    try:
        logger.info("Starting Bulk TruthGPT application...")
        
        # Import and start the application
        from main import app
        
        # Start with uvicorn
        import uvicorn
        
        config = uvicorn.Config(
            app=app,
            host=ProductionConfig.API_HOST,
            port=ProductionConfig.API_PORT,
            workers=ProductionConfig.API_WORKERS,
            log_level=ProductionConfig.LOG_LEVEL.lower(),
            access_log=True,
            reload=False
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

async def main():
    """Main startup function."""
    try:
        logger.info("Starting Bulk TruthGPT Production System...")
        
        # Check dependencies
        if not await check_dependencies():
            sys.exit(1)
        
        # Setup environment
        if not await setup_environment():
            sys.exit(1)
        
        # Start application
        await start_application()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())











