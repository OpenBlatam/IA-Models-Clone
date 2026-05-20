"""
Useful Decorators
Decorators for common functionality
"""

import functools
import time
import logging
from typing import Callable, Any
import torch

logger = logging.getLogger(__name__)


def timer(func: Callable) -> Callable:
    """Decorator to time function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.debug(f"{func.__name__} took {elapsed_time:.4f}s")
        return result
    return wrapper


def validate_inputs(validator_func: Callable = None):
    """Decorator to validate function inputs"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if validator_func:
                validator_func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def handle_errors(default_return: Any = None, log_error: bool = True):
    """Decorator to handle errors gracefully"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return default_return
        return wrapper
    return decorator


def gpu_memory_cleanup(func: Callable) -> Callable:
    """Decorator to clean up GPU memory after function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return wrapper


def no_grad(func: Callable) -> Callable:
    """Decorator to disable gradient computation"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper


def eval_mode(func: Callable) -> Callable:
    """Decorator to set model to eval mode"""
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


def train_mode(func: Callable) -> Callable:
    """Decorator to set model to train mode"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'model'):
            was_training = self.model.training
            self.model.train()
            try:
                result = func(self, *args, **kwargs)
            finally:
                if not was_training:
                    self.model.eval()
            return result
        return func(self, *args, **kwargs)
    return wrapper
