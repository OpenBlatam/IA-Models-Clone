"""
Main entry point for Contabilidad Mexicana AI SAM3
===================================================

Run the agent in continuous 24/7 mode.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from .core.contador_sam3_agent import ContadorSAM3Agent
from .config.contador_sam3_config import ContadorSAM3Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point."""
    logger.info("Starting Contabilidad Mexicana AI SAM3")
    
    # Create configuration
    config = ContadorSAM3Config()
    config.validate()
    
    # Create agent
    agent = ContadorSAM3Agent(config=config)
    
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
