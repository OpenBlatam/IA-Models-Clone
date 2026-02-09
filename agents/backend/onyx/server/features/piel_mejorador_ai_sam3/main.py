"""
Main entry point for Piel Mejorador AI SAM3
============================================

Run the agent in continuous 24/7 mode.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from .core import PielMejoradorAgent, AgentBuilder
from .config.piel_mejorador_config import PielMejoradorConfig
from .core.logging_config import setup_logging

# Setup advanced logging
logger = setup_logging(
    level="INFO",
    log_file=Path("piel_mejorador_ai_sam3.log"),
    structured=False,
    performance_tracking=True
)


async def main():
    """Main entry point."""
    logger.info("Starting Piel Mejorador AI SAM3")
    
    # Create configuration
    config = PielMejoradorConfig()
    config.validate()
    
    # Create agent using builder
    agent = (AgentBuilder()
        .with_config(config)
        .with_debug(debug)
        .build())
    
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
        await agent.stop()
        logger.info("Agent stopped")


if __name__ == "__main__":
    asyncio.run(main())

