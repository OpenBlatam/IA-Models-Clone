"""
BUL Main Entry Point
====================

Main entry point for the Business Unlimited system.
Provides CLI interface and starts the continuous processor and API server.
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path
from typing import Optional

from .config.bul_config import BULConfig
from .core.continuous_processor import ContinuousProcessor
from .api.bul_api import BULAPI
from .utils.document_processor import DocumentProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bul.log')
    ]
)

logger = logging.getLogger(__name__)

class BULSystem:
    """Main BUL system orchestrator."""
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.processor: Optional[ContinuousProcessor] = None
        self.api: Optional[BULAPI] = None
        
        # Validate configuration
        errors = self.config.validate_config()
        if errors:
            logger.error("Configuration errors:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)
        
        logger.info("BUL System initialized")
    
    async def start_processor(self):
        """Start the continuous processor."""
        if self.processor is None:
            self.processor = ContinuousProcessor(self.config)
        
        logger.info("Starting continuous processor...")
        await self.processor.start()
    
    def start_api(self, host: str = None, port: int = None):
        """Start the API server."""
        if self.api is None:
            self.api = BULAPI(self.config)
        
        logger.info("Starting API server...")
        self.api.run(host=host, port=port)
    
    async def start_full_system(self, host: str = None, port: int = None):
        """Start both processor and API server."""
        # Start processor in background
        processor_task = asyncio.create_task(self.start_processor())
        
        # Start API server (this will block)
        try:
            self.start_api(host=host, port=port)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            # Stop processor
            if self.processor:
                self.processor.stop()
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BUL - Business Unlimited System")
    parser.add_argument("--mode", choices=["processor", "api", "full"], default="full",
                       help="Run mode: processor only, API only, or both")
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = BULConfig()
    if args.debug:
        config.debug_mode = True
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create system
    system = BULSystem(config)
    
    try:
        if args.mode == "processor":
            await system.start_processor()
        elif args.mode == "api":
            system.start_api(host=args.host, port=args.port)
        else:  # full
            await system.start_full_system(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

