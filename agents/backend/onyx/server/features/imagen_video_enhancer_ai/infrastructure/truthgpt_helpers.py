"""
Helper functions for TruthGPT client operations.
"""

import logging
from typing import Dict, Any, Optional, Callable, Awaitable, TypeVar

from .truthgpt_status import TruthGPTStatus

logger = logging.getLogger(__name__)

T = TypeVar('T')


def check_truthgpt_ready(
    integration_manager: Optional[Any],
    analytics_manager: Optional[Any] = None
) -> bool:
    """
    Check if TruthGPT is ready for operations.
    
    Args:
        integration_manager: Integration manager instance
        analytics_manager: Optional analytics manager instance
        
    Returns:
        True if TruthGPT is ready, False otherwise
    """
    return (
        TruthGPTStatus.is_available() and
        integration_manager is not None and
        (analytics_manager is None or analytics_manager is not None)
    )


async def safe_truthgpt_call(
    query: str,
    operation: Callable[[], Awaitable[T]],
    fallback_value: T,
    operation_name: str = "TruthGPT operation"
) -> T:
    """
    Safely call TruthGPT operation with fallback.
    
    Args:
        query: Original query
        operation: Async operation to execute
        fallback_value: Value to return on error
        operation_name: Name of operation for logging
        
    Returns:
        Operation result or fallback value
    """
    try:
        return await operation()
    except Exception as e:
        logger.warning(f"{operation_name} failed: {e}")
        return fallback_value




