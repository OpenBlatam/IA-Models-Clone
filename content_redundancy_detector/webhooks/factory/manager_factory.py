"""
Webhook Manager Factory
Centralized creation of webhook manager instances with thread safety
"""

import logging
import threading
from typing import Optional, Dict, Any
from ..manager import WebhookManager
from ..storage import StorageFactory, StorageBackend
from ..config import WebhookConfig

logger = logging.getLogger(__name__)

# Thread-safe singleton pattern
_lock = threading.Lock()
_default_manager: Optional[WebhookManager] = None


def create_webhook_manager(
    storage_backend: Optional[StorageBackend] = None,
    enable_tracing: Optional[bool] = None,
    enable_metrics: Optional[bool] = None,
    **kwargs
) -> WebhookManager:
    """
    Create a webhook manager instance with configuration.
    
    Args:
        storage_backend: Custom storage backend (auto-created if None)
        enable_tracing: Enable OpenTelemetry tracing
        enable_metrics: Enable metrics collection
        **kwargs: Additional configuration options (max_workers, max_queue_size, etc.)
    
    Returns:
        Configured WebhookManager instance
    
    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If storage backend creation fails
    """
    try:
        # Auto-create storage backend if not provided
        if storage_backend is None:
            storage_backend = StorageFactory.create(
                storage_type=WebhookConfig.STORAGE_TYPE,
                redis_url=WebhookConfig.REDIS_URL,
                **WebhookConfig.get_redis_config()
            )
        
        # Use config defaults if not specified (early return pattern)
        if enable_tracing is None:
            enable_tracing = WebhookConfig.ENABLE_TRACING
        if enable_metrics is None:
            enable_metrics = WebhookConfig.ENABLE_METRICS
        
        manager = WebhookManager(
            storage_backend=storage_backend,
            enable_tracing=enable_tracing,
            enable_metrics=enable_metrics,
            **kwargs
        )
        
        logger.debug(
            f"WebhookManager created with tracing={enable_tracing}, "
            f"metrics={enable_metrics}, storage={type(storage_backend).__name__}"
        )
        
        return manager
        
    except Exception as e:
        logger.error(f"Failed to create WebhookManager: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create WebhookManager: {e}") from e


def get_default_webhook_manager() -> WebhookManager:
    """
    Get or create the default webhook manager instance.
    Uses thread-safe singleton pattern for global access.
    
    Returns:
        Default WebhookManager instance
    
    Note:
        Thread-safe initialization using double-checked locking pattern.
    """
    global _default_manager
    
    # Double-checked locking for thread safety
    if _default_manager is None:
        with _lock:
            if _default_manager is None:
                _default_manager = create_webhook_manager()
                logger.debug("Default WebhookManager instance created")
    
    return _default_manager


def reset_default_manager() -> None:
    """
    Reset the default manager instance.
    Useful for testing or reconfiguration.
    
    Note:
        Thread-safe operation.
    """
    global _default_manager
    
    with _lock:
        if _default_manager is not None:
            logger.debug("Resetting default WebhookManager instance")
            try:
                # Graceful shutdown if manager has cleanup
                if hasattr(_default_manager, 'stop'):
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Schedule cleanup if loop is running
                            asyncio.create_task(_default_manager.stop())
                        else:
                            loop.run_until_complete(_default_manager.stop())
                    except RuntimeError:
                        # No event loop available
                        pass
            except Exception as e:
                logger.warning(f"Error during manager cleanup: {e}")
        
        _default_manager = None

