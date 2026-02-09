"""
Graceful Shutdown
=================

Graceful shutdown handler.
"""

import logging
import asyncio
import signal
from typing import List, Callable, Optional
from fastapi import FastAPI

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Graceful shutdown handler."""
    
    def __init__(self, app: FastAPI, shutdown_timeout: float = 30.0):
        self.app = app
        self.shutdown_timeout = shutdown_timeout
        self._shutdown_handlers: List[Callable] = []
        self._shutting_down = False
    
    def register_shutdown_handler(self, handler: Callable):
        """Register shutdown handler."""
        self._shutdown_handlers.append(handler)
        logger.debug(f"Registered shutdown handler: {handler.__name__}")
    
    async def shutdown(self):
        """Execute graceful shutdown."""
        if self._shutting_down:
            return
        
        self._shutting_down = True
        logger.info("Starting graceful shutdown...")
        
        # Execute shutdown handlers
        for handler in self._shutdown_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await asyncio.wait_for(handler(), timeout=self.shutdown_timeout)
                else:
                    handler()
            except asyncio.TimeoutError:
                logger.warning(f"Shutdown handler {handler.__name__} timed out")
            except Exception as e:
                logger.error(f"Shutdown handler {handler.__name__} failed: {e}")
        
        logger.info("Graceful shutdown completed")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        logger.info("Signal handlers registered")















