"""
Component Initializer for Contador AI
======================================

Centralizes conditional initialization of optional components.
Eliminates duplicate try/except ImportError patterns.

Single Responsibility: Handle conditional initialization of optional components.
"""

import logging
from typing import Optional, TypeVar, Callable, Any

logger = logging.getLogger(__name__)

T = TypeVar('T')


def initialize_optional_component(
    module_path: str,
    class_name: str,
    init_args: Optional[tuple] = None,
    init_kwargs: Optional[dict] = None,
    warning_message: Optional[str] = None,
    default_value: Any = None
) -> Optional[Any]:
    """
    Initialize an optional component with graceful fallback.
    
    Args:
        module_path: Path to the module (e.g., ".cache_manager")
        class_name: Name of the class to instantiate
        init_args: Optional positional arguments for initialization
        init_kwargs: Optional keyword arguments for initialization
        warning_message: Custom warning message (defaults to "{class_name} not available")
        default_value: Value to return if initialization fails (defaults to None)
        
    Returns:
        Initialized component instance or default_value if initialization fails
        
    Examples:
        >>> cache = initialize_optional_component(
        ...     ".cache_manager",
        ...     "ResponseCache",
        ...     init_kwargs={"max_size": 1000, "default_ttl": 3600},
        ...     warning_message="Cache manager not available"
        ... )
    """
    try:
        module = __import__(module_path, fromlist=[class_name], level=1)
        component_class = getattr(module, class_name)
        
        init_args = init_args or ()
        init_kwargs = init_kwargs or {}
        
        return component_class(*init_args, **init_kwargs)
    except ImportError as e:
        warning = warning_message or f"{class_name} not available"
        logger.warning(f"{warning}: {e}")
        return default_value
    except Exception as e:
        warning = warning_message or f"Error initializing {class_name}"
        logger.warning(f"{warning}: {e}")
        return default_value


def initialize_cache(config: Any) -> Optional[Any]:
    """
    Initialize cache component if enabled in config.
    
    Args:
        config: Configuration object with cache_enabled and cache_ttl attributes
        
    Returns:
        Cache instance or None if disabled/unavailable
    """
    if not getattr(config, 'cache_enabled', False):
        return None
    
    return initialize_optional_component(
        module_path=".cache_manager",
        class_name="ResponseCache",
        init_kwargs={
            "max_size": 1000,
            "default_ttl": getattr(config, 'cache_ttl', 3600)
        },
        warning_message="Cache manager not available"
    )


def initialize_metrics() -> Optional[Any]:
    """
    Initialize metrics collector component.
    
    Returns:
        MetricsCollector instance or None if unavailable
    """
    return initialize_optional_component(
        module_path=".metrics_collector",
        class_name="MetricsCollector",
        warning_message="Metrics collector not available"
    )

