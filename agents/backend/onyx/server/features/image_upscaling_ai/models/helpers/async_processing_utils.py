"""
Async Processing Utilities
==========================

Utilities for asynchronous image processing.
"""

import logging
import asyncio
from typing import Callable, Optional, Union, Tuple, Any
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from .metrics_utils import UpscalingMetrics

logger = logging.getLogger(__name__)


class AsyncProcessingUtils:
    """Utilities for asynchronous processing."""
    
    @staticmethod
    async def run_async(
        executor: ThreadPoolExecutor,
        func: Callable,
        *args,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> Any:
        """
        Run a function asynchronously in an executor.
        
        Args:
            executor: ThreadPoolExecutor instance
            func: Function to run
            *args: Positional arguments for func
            progress_callback: Optional callback for progress updates
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func execution
        """
        loop = asyncio.get_event_loop()
        
        # Wrap progress callback for async
        async_progress = None
        if progress_callback:
            def async_callback(current, total):
                loop.call_soon_threadsafe(progress_callback, current, total)
            async_progress = async_callback
        
        return await loop.run_in_executor(
            executor,
            lambda: func(*args, progress_callback=async_progress, **kwargs)
        )


