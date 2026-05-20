"""
GPU Error Handling Utilities
=============================

Helper functions for handling GPU-related errors consistently.
This eliminates repetitive error handling code throughout the codebase.

Benefits:
- Consistent error handling across all GPU operations
- Centralized memory cleanup logic
- Better error messages for users
- Easier to update error handling behavior
"""

import logging
import gc
from typing import Callable, TypeVar, Optional, Any
import torch

logger = logging.getLogger(__name__)

# Type variable for return type
T = TypeVar('T')


def handle_gpu_errors(
    operation: Callable[[], T],
    operation_name: str = "GPU operation",
    cleanup_on_error: bool = True,
    reraise: bool = True,
) -> T:
    """
    Execute a GPU operation with consistent error handling.
    
    This helper wraps GPU operations to provide consistent error handling
    for out-of-memory errors and other GPU-related exceptions.
    
    Args:
        operation: Callable that performs the GPU operation
        operation_name: Name of the operation for logging
        cleanup_on_error: Whether to clear CUDA cache on error
        reraise: Whether to re-raise the exception after handling
    
    Returns:
        Result of the operation
    
    Raises:
        RuntimeError: If operation fails with GPU-related error
    
    Example:
        >>> def generate_image():
        ...     # GPU operation here
        ...     return result
        >>> 
        >>> result = handle_gpu_errors(
        ...     generate_image,
        ...     operation_name="Image generation"
        ... )
    """
    try:
        return operation()
    
    except RuntimeError as e:
        error_str = str(e).lower()
        if "out of memory" in error_str or "cuda" in error_str:
            logger.error(f"GPU out of memory during {operation_name}")
            
            if cleanup_on_error:
                clear_gpu_memory()
            
            if reraise:
                raise RuntimeError(
                    f"GPU memory insufficient during {operation_name}. "
                    "Try reducing resolution, quality, or batch size."
                ) from e
            
            return None
    
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA out of memory error during {operation_name}")
        
        if cleanup_on_error:
            clear_gpu_memory()
        
        if reraise:
            raise RuntimeError(
                f"GPU memory insufficient during {operation_name}. "
                "Try reducing resolution, quality, or batch size."
            ) from e
        
        return None
    
    except Exception as e:
        logger.error(f"{operation_name} failed: {e}", exc_info=True)
        if reraise:
            raise RuntimeError(f"{operation_name} failed: {e}") from e
        return None


def clear_gpu_memory() -> None:
    """
    Clear GPU memory cache and run garbage collection.
    
    This helper centralizes memory cleanup logic that was duplicated
    across multiple files (avatar_manager.py, voice_engine.py, etc.).
    
    Example:
        >>> try:
        ...     result = some_gpu_operation()
        ... except RuntimeError:
        ...     clear_gpu_memory()
        ...     raise
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared")
    
    gc.collect()
    logger.debug("Garbage collection completed")


def safe_gpu_operation(
    operation: Callable[[], T],
    fallback: Optional[Callable[[], T]] = None,
    operation_name: str = "GPU operation",
) -> T:
    """
    Execute GPU operation with automatic fallback on error.
    
    Args:
        operation: Primary GPU operation
        fallback: Optional fallback operation if primary fails
        operation_name: Name of operation for logging
    
    Returns:
        Result of operation or fallback
    
    Example:
        >>> def high_quality_generate():
        ...     # High quality but memory-intensive
        ...     return result
        >>> 
        >>> def low_quality_generate():
        ...     # Lower quality but memory-efficient
        ...     return result
        >>> 
        >>> result = safe_gpu_operation(
        ...     high_quality_generate,
        ...     fallback=low_quality_generate,
        ...     operation_name="Image generation"
        ... )
    """
    try:
        return handle_gpu_errors(operation, operation_name, reraise=False)
    except RuntimeError:
        if fallback:
            logger.warning(
                f"{operation_name} failed, attempting fallback operation"
            )
            return handle_gpu_errors(fallback, f"{operation_name} (fallback)")
        raise








