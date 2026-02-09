"""
Main entry point for Imagen Video Enhancer AI
=============================================

Run the agent in continuous 24/7 mode.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from .core.enhancer_agent import EnhancerAgent
from .config.enhancer_config import EnhancerConfig
from .core.logging_config import setup_logging

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


async def main():
    """Main entry point."""
    logger.info("Starting Imagen Video Enhancer AI")
    
    # Create configuration
    config = EnhancerConfig()
    config.validate()
    
    # Create agent
    agent = EnhancerAgent(config=config)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down...")
        asyncio.create_task(agent.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start agent in continuous mode
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
    finally:
        await agent.close()
        logger.info("Agent stopped")


if __name__ == "__main__":
    asyncio.run(main())

