"""
Main Entry Point for Unified AI History Comparison System

This is the main entry point for the completely refactored and unified
AI History Comparison System that integrates all advanced features.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to the path to import core modules
sys.path.append(str(Path(__file__).parent.parent))

from unified_config import UnifiedConfig, get_config
from unified_manager import UnifiedSystemManager, get_unified_manager
from unified_api import app

logger = logging.getLogger(__name__)

async def main():
    """Main function to run the unified system"""
    try:
        # Load configuration
        config = get_config()
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.api.log_level.value),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        logger.info("Starting Unified AI History Comparison System...")
        logger.info(f"Configuration: {config.get_summary()}")
        
        # Initialize unified manager
        manager = get_unified_manager()
        success = await manager.initialize()
        
        if not success:
            logger.error("Failed to initialize unified system")
            return 1
        
        logger.info("Unified system initialized successfully")
        
        # Get system status
        status = manager.get_system_status()
        logger.info(f"System status: {status}")
        
        # Run the FastAPI application
        import uvicorn
        uvicorn.run(
            app,
            host=config.api.host,
            port=config.api.port,
            workers=config.api.workers,
            reload=config.api.reload,
            log_level=config.api.log_level.value.lower()
        )
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        return 0
    except Exception as e:
        logger.error(f"Failed to start unified system: {e}")
        return 1
    finally:
        # Cleanup
        try:
            manager = get_unified_manager()
            await manager.shutdown()
            logger.info("Unified system shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))





















