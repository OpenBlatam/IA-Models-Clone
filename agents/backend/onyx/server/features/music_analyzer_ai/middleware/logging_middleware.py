"""
Logging Middleware - Log requests and responses
"""

from typing import Any, Callable
import logging
import time

from .middleware import BaseMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseMiddleware):
    """
    Middleware that logs requests and responses
    """
    
    def __init__(self, log_level: int = logging.INFO, log_response: bool = False):
        super().__init__("LoggingMiddleware")
        self.log_level = log_level
        self.log_response = log_response
    
    def process(self, request: Any, next_handler: Callable) -> Any:
        """Process with logging"""
        start_time = time.time()
        
        # Log request
        logger.log(self.log_level, f"Processing request: {type(request).__name__}")
        
        try:
            # Process
            response = next_handler(request)
            
            # Log response
            elapsed = time.time() - start_time
            logger.log(
                self.log_level,
                f"Request processed in {elapsed:.3f}s"
            )
            
            if self.log_response:
                logger.debug(f"Response: {response}")
            
            return response
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Request failed after {elapsed:.3f}s: {str(e)}",
                exc_info=True
            )
            raise








