"""
Retry Module - DEPRECATED

This module is deprecated. Please use retry_utils instead:

    from core.retry_utils import RetryStrategy, RetryPolicy, retry, RetryExecutor

This file is kept for backward compatibility only and re-exports from retry_utils.
"""

import warnings

warnings.warn(
    "core.retry is deprecated. Use core.retry_utils instead. "
    "This file will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from resilience.retry_utils
from .resilience.retry_utils import (
    RetryStrategy,
    RetryPolicy,
    RetryResult,
    RetryExecutor,
    retry,
    retry_context,
    execute_with_retry,
    RetryManager,
    get_retry_manager,
)

__all__ = [
    "RetryStrategy",
    "RetryPolicy",
    "RetryResult",
    "RetryExecutor",
    "retry",
    "retry_context",
    "execute_with_retry",
    "RetryManager",
    "get_retry_manager",
]
