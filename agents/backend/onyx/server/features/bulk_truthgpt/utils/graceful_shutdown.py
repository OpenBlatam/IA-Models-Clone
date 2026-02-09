"""
Graceful Shutdown
=================

Advanced graceful shutdown handling for clean service termination.
"""

import asyncio
import logging
import signal
from typing import List, Callable, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class GracefulShutdown:
    """Advanced graceful shutdown manager."""
    
    def __init__(self, shutdown_timeout: float = 30.0):
        self.shutdown_timeout = shutdown_timeout
        self.shutdown_handlers: List[Callable] = []
        self.is_shutting_down = False
        self.shutdown_started_at: Optional[datetime] = None
    
    def register_handler(self, handler: Callable, priority: int = 0):
        """Register a shutdown handler."""
        self.shutdown_handlers.append((priority, handler))
        self.shutdown_handlers.sort(key=lambda x: x[0], reverse=True)
        logger.debug(f"Shutdown handler registered: {handler.__name__}")
    
    async def shutdown(self):
        """Perform graceful shutdown."""
        if self.is_shutting_down:
            logger.warning("Shutdown already in progress")
            return
        
        self.is_shutting_down = True
        self.shutdown_started_at = datetime.now()
        
        logger.info("Starting graceful shutdown...")
        
        try:
            # Execute all shutdown handlers
            for priority, handler in self.shutdown_handlers:
                try:
                    logger.info(f"Executing shutdown handler: {handler.__name__}")
                    
                    if asyncio.iscoroutinefunction(handler):
                        await asyncio.wait_for(
                            handler(),
                            timeout=self.shutdown_timeout
                        )
                    else:
                        # Run in executor for sync functions
                        loop = asyncio.get_event_loop()
                        await asyncio.wait_for(
                            loop.run_in_executor(None, handler),
                            timeout=self.shutdown_timeout
                        )
                    
                    logger.info(f"Shutdown handler completed: {handler.__name__}")
                    
                except asyncio.TimeoutError:
                    logger.error(f"Shutdown handler timeout: {handler.__name__}")
                except Exception as e:
                    logger.error(f"Shutdown handler error: {handler.__name__} - {e}")
            
            shutdown_duration = (datetime.now() - self.shutdown_started_at).total_seconds()
            logger.info(f"Graceful shutdown completed in {shutdown_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("Signal handlers registered for graceful shutdown")
    
    def get_stats(self) -> dict:
        """Get shutdown statistics."""
        return {
            "is_shutting_down": self.is_shutting_down,
            "shutdown_timeout": self.shutdown_timeout,
            "registered_handlers": len(self.shutdown_handlers),
            "shutdown_started_at": self.shutdown_started_at.isoformat() if self.shutdown_started_at else None
        }

# Global instance
graceful_shutdown = GracefulShutdown(shutdown_timeout=30.0)
















