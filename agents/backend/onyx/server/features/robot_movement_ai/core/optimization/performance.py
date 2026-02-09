"""
Performance utilities for trajectory optimization
"""

import time
from functools import wraps
from typing import Callable, Any


def measure_time(func: Callable) -> Callable:
    """Measure execution time of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper



