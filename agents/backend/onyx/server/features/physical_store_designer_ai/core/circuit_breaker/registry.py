"""
Circuit Breaker Registry

Registry and decorator for circuit breakers.
Provides global registry and decorator functionality.
"""

from typing import Callable, Any, Optional, Dict
from functools import wraps
import asyncio

from .breaker import CircuitBreaker
from .config import CircuitBreakerConfig


# Global circuit breakers registry (thread-safe)
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = asyncio.Lock()


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
    name: Optional[str] = None,
    config: Optional[CircuitBreakerConfig] = None
):
    """
    Decorator to add circuit breaker to functions
    
    Args:
        failure_threshold: Number of failures before opening circuit (deprecated, use config)
        recovery_timeout: Seconds to wait before attempting recovery (deprecated, use config)
        expected_exception: Exception type that triggers circuit breaker (deprecated, use config)
        name: Name for the circuit breaker (defaults to function name)
        config: Circuit breaker configuration (takes precedence over individual params)
    """
    def decorator(func: Callable) -> Callable:
        cb_name = name or f"{func.__module__}.{func.__name__}"
        
        # Support both old and new API
        if config is None:
            breaker = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                name=cb_name
            )
        else:
            breaker = CircuitBreaker(config=config, name=cb_name)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run in executor
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we can't use run_until_complete
                    # Schedule and wait using a future
                    import concurrent.futures
                    future = asyncio.run_coroutine_threadsafe(
                        breaker.call(func, *args, **kwargs),
                        loop
                    )
                    return future.result()
                else:
                    return loop.run_until_complete(
                        breaker.call(func, *args, **kwargs)
                    )
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(breaker.call(func, *args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


async def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name (thread-safe)
    
    Args:
        name: Name of the circuit breaker
        **kwargs: Configuration parameters (passed to CircuitBreaker constructor)
    
    Returns:
        CircuitBreaker instance
    """
    async with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
        return _circuit_breakers[name]


def get_circuit_breaker_sync(name: str, **kwargs) -> CircuitBreaker:
    """
    Synchronous version of get_circuit_breaker for use in non-async contexts
    
    Args:
        name: Name of the circuit breaker
        **kwargs: Configuration parameters (passed to CircuitBreaker constructor)
    
    Returns:
        CircuitBreaker instance
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we can't use run_until_complete
            # Create directly without lock (not ideal but works)
            if name not in _circuit_breakers:
                _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
            return _circuit_breakers[name]
        else:
            return loop.run_until_complete(get_circuit_breaker(name, **kwargs))
    except RuntimeError:
        # No event loop, create directly
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
        return _circuit_breakers[name]


def get_all_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """Get state of all circuit breakers"""
    return {name: cb.get_state() for name, cb in _circuit_breakers.items()}


def reset_all_circuit_breakers():
    """Reset all circuit breakers"""
    async def _reset_all():
        async with _registry_lock:
            for breaker in _circuit_breakers.values():
                await breaker.reset()
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_reset_all())
        else:
            loop.run_until_complete(_reset_all())
    except RuntimeError:
        # No event loop, reset directly (not thread-safe but better than nothing)
        for breaker in _circuit_breakers.values():
            asyncio.run(breaker.reset())




