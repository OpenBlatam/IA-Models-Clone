#!/usr/bin/env python3
"""
Main entry point for the integrated OS Content system
Runs all optimized components with clean architecture
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Any, List, Dict, Optional, Union, Tuple
from typing_extensions import Literal, TypedDict

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the integrated application
from integrated_app import app
from refactored_architecture import RefactoredOSContentApplication
import uvicorn

# Configure logging with better formatting and handlers
def setup_logging():
    """Configure comprehensive logging system"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / 'os_content.log'),
            logging.FileHandler(logs_dir / 'os_content_error.log', level=logging.ERROR)
        ]
    )
    
    # Set specific log levels for external libraries
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

async def main():
    """Main application entry point"""
    app_instance = None
    
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("🚀 Starting OS Content System...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Create and initialize the application
        app_instance = RefactoredOSContentApplication()
        await app_instance.initialize()
        
        logger.info("✅ OS Content System initialized successfully")
        logger.info("🌐 Starting FastAPI server on http://0.0.0.0:8000")
        logger.info("📚 API documentation available at http://0.0.0.0:8000/docs")
        logger.info("🔍 Health check available at http://0.0.0.0:8000/health")
        
        # Run the FastAPI application with optimized configuration
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            reload=False,
            workers=1,
            loop="asyncio",
            http="httptools",
            ws="websockets"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down OS Content System...")
    except Exception as e:
        logger.error(f"❌ Error running OS Content System: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if app_instance:
            try:
                await app_instance.shutdown()
                logger.info("🔄 Application shutdown completed")
            except Exception as e:
                logger.error(f"⚠️ Error during shutdown: {e}")
        logger.info("🛑 OS Content System stopped")

if __name__ == "__main__":
    asyncio.run(main()) 