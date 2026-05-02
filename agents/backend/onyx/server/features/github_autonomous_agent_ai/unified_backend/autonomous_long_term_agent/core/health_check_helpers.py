"""
Health Check Helpers
Utility functions for health checking operations
"""

import logging
from typing import Callable, Awaitable, Any
from .health_check import HealthCheck, HealthStatus

logger = logging.getLogger(__name__)


async def execute_health_check(
    check_name: str,
    check_function: Callable[[], Awaitable[HealthCheck]]
) -> HealthCheck:
    """
    Execute a health check with consistent error handling.
    Centralizes the try/except pattern used in all health checks.
    
    Args:
        check_name: Name of the health check
        check_function: Async function that performs the check and returns HealthCheck
        
    Returns:
        HealthCheck result, or HealthCheck with UNKNOWN status on error
    """
    try:
        return await check_function()
    except Exception as e:
        logger.error(f"Error checking {check_name}: {e}", exc_info=True)
        return HealthCheck(
            name=check_name,
            status=HealthStatus.UNKNOWN,
            message=f"Error checking {check_name}: {e}",
            details={"error": str(e)}
        )

