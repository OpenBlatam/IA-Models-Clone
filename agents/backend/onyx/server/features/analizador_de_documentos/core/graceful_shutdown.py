"""
Graceful Shutdown Manager for Document Analyzer
================================================

Advanced graceful shutdown handling with cleanup and resource management.
"""

import asyncio
import logging
import signal
from typing import Dict, List, Callable, Optional
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ShutdownHandler:
    """Shutdown handler definition"""
    name: str
    handler: Callable
    priority: int = 0  # Lower priority = executed first
    timeout: float = 5.0

class GracefulShutdown:
    """Advanced graceful shutdown manager"""
    
    def __init__(self):
        self.handlers: List[ShutdownHandler] = []
        self.is_shutting_down = False
        self.shutdown_started: Optional[datetime] = None
        logger.info("GracefulShutdown initialized")
    
    def register_handler(
        self,
        handler: Callable,
        name: str = "unknown",
        priority: int = 0,
        timeout: float = 5.0
    ):
        """Register a shutdown handler"""
        shutdown_handler = ShutdownHandler(
            name=name,
            handler=handler,
            priority=priority,
            timeout=timeout
        )
        self.handlers.append(shutdown_handler)
        # Sort by priority
        self.handlers.sort(key=lambda x: x.priority)
        logger.info(f"Registered shutdown handler: {name} (priority: {priority})")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("Signal handlers registered")
    
    async def shutdown(self, timeout: float = 30.0):
        """Execute graceful shutdown"""
        if self.is_shutting_down:
            logger.warning("Shutdown already in progress")
            return
        
        self.is_shutting_down = True
        self.shutdown_started = datetime.now()
        
        logger.info(f"Starting graceful shutdown with {len(self.handlers)} handlers")
        
        # Execute handlers in priority order
        for handler_info in self.handlers:
            try:
                logger.info(f"Executing shutdown handler: {handler_info.name}")
                
                # Execute with timeout
                if asyncio.iscoroutinefunction(handler_info.handler):
                    await asyncio.wait_for(
                        handler_info.handler(),
                        timeout=handler_info.timeout
                    )
                else:
                    await asyncio.wait_for(
                        asyncio.to_thread(handler_info.handler),
                        timeout=handler_info.timeout
                    )
                
                logger.info(f"Completed shutdown handler: {handler_info.name}")
            except asyncio.TimeoutError:
                logger.error(f"Shutdown handler {handler_info.name} timed out")
            except Exception as e:
                logger.error(f"Error in shutdown handler {handler_info.name}: {e}")
        
        duration = (datetime.now() - self.shutdown_started).total_seconds()
        logger.info(f"Graceful shutdown completed in {duration:.2f}s")

# Global instance
graceful_shutdown = GracefulShutdown()
















