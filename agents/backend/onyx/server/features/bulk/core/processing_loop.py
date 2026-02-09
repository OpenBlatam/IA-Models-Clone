"""
Processing Loop - Common processing loop utilities
===================================================

Provides common utilities for running processing loops.
"""

import asyncio
import logging
from typing import Callable, Optional, Any

from .constants import DEFAULT_LOOP_SLEEP_SECONDS

logger = logging.getLogger(__name__)


class ProcessingLoop:
    """Common utilities for processing loops."""
    
    @staticmethod
    async def run_processing_loop(
        processor: Any,
        process_func: Callable,
        update_func: Optional[Callable] = None,
        sleep_seconds: float = DEFAULT_LOOP_SLEEP_SECONDS,
        loop_name: str = "processing"
    ) -> None:
        """
        Run a standard processing loop.
        
        Args:
            processor: Processor instance with is_running attribute
            process_func: Async function to call in each iteration
            update_func: Optional async function to call for updates
            sleep_seconds: Seconds to sleep between iterations
            loop_name: Name of the loop for logging
        """
        if processor.is_running:
            logger.warning(f"{loop_name} is already running")
            return
        
        processor.is_running = True
        logger.info(f"Starting {loop_name} loop...")
        
        try:
            while processor.is_running:
                await process_func()
                
                if update_func:
                    await update_func()
                
                await asyncio.sleep(sleep_seconds)
                
        except Exception as e:
            logger.error(f"Error in {loop_name} loop: {e}", exc_info=True)
            if hasattr(processor, 'callbacks'):
                await processor.callbacks.execute_error_callback(e)
        finally:
            processor.is_running = False
            logger.info(f"{loop_name} loop stopped")






