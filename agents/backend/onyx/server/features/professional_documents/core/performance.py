"""
Performance monitoring utilities.

Context managers and utilities for measuring execution time.
"""

import time
import logging
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


@contextmanager
def measure_time(operation_name: str) -> Generator[None, None, None]:
    """
    Context manager to measure execution time of an operation.
    
    Args:
        operation_name: Name of the operation being measured
        
    Yields:
        None
        
    Example:
        with measure_time("document_generation"):
            # Your code here
            pass
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        logger.debug(f"{operation_name} took {elapsed_time:.3f} seconds")


class PerformanceTimer:
    """Context manager for measuring performance with detailed logging."""
    
    def __init__(self, operation_name: str, log_level: int = logging.INFO):
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time: float = 0.0
        self.elapsed_time: float = 0.0
    
    def __enter__(self) -> "PerformanceTimer":
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.start_time
        logger.log(
            self.log_level,
            f"{self.operation_name} completed in {self.elapsed_time:.3f} seconds"
        )
        return False
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time if self.elapsed_time == 0 else self.elapsed_time






