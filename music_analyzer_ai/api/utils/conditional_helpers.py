"""
Conditional operation helper functions.

This module provides utilities for executing operations conditionally
based on parameters, flags, or service availability.
"""

from typing import Any, Callable, Optional, Dict, Awaitable
import logging

logger = logging.getLogger(__name__)


async def execute_if_condition(
    condition: bool,
    operation: Callable[..., Awaitable[Any]],
    *args,
    on_false: Optional[Callable] = None,
    **kwargs
) -> Optional[Any]:
    """
    Execute an async operation only if condition is True.
    
    Args:
        condition: Boolean condition to check
        operation: Async function to execute if condition is True
        *args: Positional arguments for operation
        on_false: Optional function to execute if condition is False
        **kwargs: Keyword arguments for operation
    
    Returns:
        Result of operation if condition is True, None or on_false result otherwise
    
    Example:
        result = await execute_if_condition(
            request.include_coaching,
            music_coach.generate_coaching_analysis,
            analysis
        )
    """
    if condition:
        return await operation(*args, **kwargs)
    elif on_false:
        return await on_false(*args, **kwargs)
    return None


def execute_if_condition_sync(
    condition: bool,
    operation: Callable[..., Any],
    *args,
    on_false: Optional[Callable] = None,
    **kwargs
) -> Optional[Any]:
    """
    Execute a sync operation only if condition is True.
    
    Args:
        condition: Boolean condition to check
        operation: Function to execute if condition is True
        *args: Positional arguments for operation
        on_false: Optional function to execute if condition is False
        **kwargs: Keyword arguments for operation
    
    Returns:
        Result of operation if condition is True, None or on_false result otherwise
    """
    if condition:
        return operation(*args, **kwargs)
    elif on_false:
        return on_false(*args, **kwargs)
    return None


async def execute_with_service(
    service: Optional[Any],
    method_name: str,
    *args,
    log_missing: bool = True,
    **kwargs
) -> Optional[Any]:
    """
    Execute a service method only if service is available.
    
    Args:
        service: Service instance (may be None)
        method_name: Name of method to call on service
        *args: Positional arguments for method
        log_missing: Whether to log if service is missing
        **kwargs: Keyword arguments for method
    
    Returns:
        Result of method call or None if service is missing
    
    Example:
        await execute_with_service(
            webhook_service,
            "trigger_webhook",
            event_type,
            data
        )
    """
    if service is None:
        if log_missing:
            logger.debug(f"Service not available, skipping {method_name}")
        return None
    
    if not hasattr(service, method_name):
        logger.warning(f"Service {type(service)} does not have method {method_name}")
        return None
    
    method = getattr(service, method_name)
    
    # Check if method is async
    import inspect
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    else:
        return method(*args, **kwargs)


async def execute_multiple_conditionally(
    operations: list[tuple[bool, Callable, tuple, dict]],
    continue_on_error: bool = True
) -> list[Optional[Any]]:
    """
    Execute multiple operations conditionally.
    
    Args:
        operations: List of tuples (condition, operation, args, kwargs)
        continue_on_error: Whether to continue if one operation fails
    
    Returns:
        List of results (None for skipped or failed operations)
    
    Example:
        results = await execute_multiple_conditionally([
            (request.include_coaching, music_coach.generate_coaching, (analysis,), {}),
            (request.save_history, history_service.add_analysis, (track_id, analysis), {}),
        ])
    """
    import inspect
    results = []
    
    for condition, operation, args, kwargs in operations:
        if not condition:
            results.append(None)
            continue
        
        try:
            if inspect.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            results.append(result)
        except Exception as e:
            if continue_on_error:
                logger.warning(f"Conditional operation failed: {e}")
                results.append(None)
            else:
                raise
    
    return results


def apply_if_not_none(
    value: Optional[Any],
    operation: Callable[[Any], Any],
    default: Any = None
) -> Any:
    """
    Apply an operation to a value only if it's not None.
    
    Args:
        value: Value to check and potentially transform
        operation: Function to apply to value
        default: Default value if value is None
    
    Returns:
        Result of operation(value) or default
    
    Example:
        album_name = apply_if_not_none(
            track.get("album"),
            lambda a: a.get("name") if isinstance(a, dict) else a,
            default="Unknown"
        )
    """
    if value is not None:
        return operation(value)
    return default


async def apply_if_not_none_async(
    value: Optional[Any],
    operation: Callable[[Any], Awaitable[Any]],
    default: Any = None
) -> Any:
    """
    Apply an async operation to a value only if it's not None.
    
    Args:
        value: Value to check and potentially transform
        operation: Async function to apply to value
        default: Default value if value is None
    
    Returns:
        Result of await operation(value) or default
    """
    if value is not None:
        return await operation(value)
    return default

