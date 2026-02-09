"""
Timeout Module - DEPRECATED

This module is deprecated. Please use timeout_utils instead:

    from core.timeout_utils import TimeoutException, TimeoutManager, with_timeout

This file is kept for backward compatibility only and re-exports from timeout_utils.
"""

import warnings

warnings.warn(
    "core.timeout is deprecated. Use core.timeout_utils instead. "
    "This file will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from resilience.timeout_utils
from .resilience.timeout_utils import (
    TimeoutException,
    TimeoutManager,
    timeout_context,
    with_timeout,
    execute_with_timeout,
    get_timeout_manager,
)

__all__ = [
    "TimeoutException",
    "TimeoutManager",
    "timeout_context",
    "with_timeout",
    "execute_with_timeout",
    "get_timeout_manager",
]
