"""
Main entry point for Color Grading AI TruthGPT
===============================================

Run the agent for color grading operations.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from .core.color_grading_agent import ColorGradingAgent
from .config.color_grading_config import ColorGradingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point."""
    logger.info("Starting Color Grading AI TruthGPT")
    
    # Create configuration
    config = ColorGradingConfig()
    config.validate()
    
    # Create agent
    agent = ColorGradingAgent(config=config)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down...")
        asyncio.create_task(agent.close())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Example usage
        logger.info("Color Grading AI TruthGPT ready")
        logger.info("Use the agent methods to process videos and images")
        
        # Keep running (in production, would have continuous loop)
        await asyncio.sleep(3600)  # Run for 1 hour as example
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
    finally:
        await agent.close()
        logger.info("Agent stopped")


if __name__ == "__main__":
    asyncio.run(main())




