"""
Signal Handler - Graceful shutdown handling
============================================

Handles system signals for graceful shutdown of the continuous processor.
"""

import logging
import signal
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


class SignalHandler:
    """Handles system signals for graceful shutdown."""
    
    def __init__(self, shutdown_callback: Optional[Callable[[], None]] = None):
        self.shutdown_callback = shutdown_callback
        self._original_handlers = {}
        self._is_setup = False
    
    def setup(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        if self._is_setup:
            logger.warning("Signal handlers already setup")
            return
        
        def signal_handler(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            if self.shutdown_callback:
                self.shutdown_callback()
        
        try:
            self._original_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, signal_handler)
            self._original_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, signal_handler)
            self._is_setup = True
            logger.debug("Signal handlers setup complete")
        except (ValueError, OSError) as e:
            logger.warning(f"Could not setup signal handlers: {e}")
    
    def restore(self) -> None:
        """Restore original signal handlers."""
        if not self._is_setup:
            return
        
        try:
            for sig, handler in self._original_handlers.items():
                if handler is not None:
                    signal.signal(sig, handler)
            self._is_setup = False
            logger.debug("Signal handlers restored")
        except (ValueError, OSError) as e:
            logger.warning(f"Could not restore signal handlers: {e}")






