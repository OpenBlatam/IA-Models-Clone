"""
Decorators
Useful decorators for ML operations.
"""

import functools
import time
import logging
from typing import Callable, Any, Optional
import torch

logger = logging.getLogger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{func.__name__} took {duration:.4f} seconds")
        return result
    return wrapper


def gpu_memory_tracker(func: Callable) -> Callable:
    """Decorator to track GPU memory usage."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            logger.info(
                f"{func.__name__} - Memory: {start_memory:.2f}MB -> {end_memory:.2f}MB, "
                f"Peak: {peak_memory:.2f}MB"
            )
        
        return result
    return wrapper


def error_handler(error_message: Optional[str] = None, default_return: Any = None):
    """
    Decorator for error handling.
    
    Args:
        error_message: Custom error message
        default_return: Value to return on error
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                msg = error_message or f"Error in {func.__name__}"
                logger.error(f"{msg}: {e}", exc_info=True)
                return default_return
        return wrapper
    return decorator


def validate_inputs(**validators):
    """
    Decorator to validate function inputs.
    
    Args:
        **validators: Dictionary mapping parameter names to validation functions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Validation failed for {param_name}: {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def cache_result(cache_key_func: Optional[Callable] = None, ttl: Optional[int] = None):
    """
    Decorator to cache function results.
    
    Args:
        cache_key_func: Function to generate cache key from arguments
        ttl: Time to live in seconds
    """
    cache = {}
    cache_times = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                key = cache_key_func(*args, **kwargs)
            else:
                key = str(args) + str(sorted(kwargs.items()))
            
            # Check cache
            if key in cache:
                if ttl is None:
                    return cache[key]
                elif time.time() - cache_times[key] < ttl:
                    return cache[key]
                else:
                    # Expired
                    del cache[key]
                    del cache_times[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache[key] = result
            cache_times[key] = time.time()
            
            return result
        return wrapper
    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry function on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay:.2f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            
            raise last_exception
        return wrapper
    return decorator


def torch_no_grad(func: Callable) -> Callable:
    """Decorator to disable gradient computation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper


def torch_eval_mode(func: Callable) -> Callable:
    """Decorator to set model to eval mode."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'model'):
            was_training = self.model.training
            self.model.eval()
            try:
                result = func(self, *args, **kwargs)
            finally:
                if was_training:
                    self.model.train()
            return result
        return func(self, *args, **kwargs)
    return wrapper



