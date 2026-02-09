"""
Middleware registry for centralized middleware management.

Allows dynamic registration and application of middleware.
"""

import logging
from typing import Dict, List, Type, Optional, Any
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class MiddlewareRegistry:
    """
    Centralized registry for managing middleware.
    
    Provides a clean way to register and apply middleware to FastAPI apps.
    """
    
    def __init__(self):
        self._middleware: Dict[str, Dict[str, Any]] = {}
        logger.info("MiddlewareRegistry initialized")
    
    def register(
        self,
        name: str,
        middleware_class: Type[BaseHTTPMiddleware],
        priority: int = 0,
        **kwargs
    ):
        """
        Register middleware with the registry.
        
        Args:
            name: Unique name for the middleware
            middleware_class: Middleware class to register
            priority: Priority order (lower = earlier in stack)
            **kwargs: Additional configuration for middleware
        """
        if name in self._middleware:
            logger.warning(f"Middleware '{name}' already registered. Overwriting.")
        
        self._middleware[name] = {
            "class": middleware_class,
            "priority": priority,
            "config": kwargs
        }
        logger.debug(f"Middleware '{name}' registered with priority {priority}")
    
    def apply_all(self, app: FastAPI) -> None:
        """
        Apply all registered middleware to the FastAPI app.
        
        Middleware is applied in priority order (lowest first).
        
        Args:
            app: FastAPI application instance
        """
        # Sort by priority
        sorted_middleware = sorted(
            self._middleware.items(),
            key=lambda x: x[1]["priority"]
        )
        
        for name, middleware_info in sorted_middleware:
            try:
                middleware_class = middleware_info["class"]
                config = middleware_info["config"]
                
                app.add_middleware(middleware_class, **config)
                logger.debug(f"Applied middleware: {name}")
            except Exception as e:
                logger.error(f"Error applying middleware '{name}': {e}", exc_info=True)
        
        logger.info(f"Applied {len(sorted_middleware)} middleware to app")
    
    def apply_selected(
        self,
        app: FastAPI,
        middleware_names: List[str]
    ) -> None:
        """
        Apply only selected middleware to the FastAPI app.
        
        Args:
            app: FastAPI application instance
            middleware_names: List of middleware names to apply
        """
        # Filter and sort by priority
        selected = [
            (name, info)
            for name, info in self._middleware.items()
            if name in middleware_names
        ]
        selected.sort(key=lambda x: x[1]["priority"])
        
        for name, middleware_info in selected:
            try:
                middleware_class = middleware_info["class"]
                config = middleware_info["config"]
                
                app.add_middleware(middleware_class, **config)
                logger.debug(f"Applied middleware: {name}")
            except Exception as e:
                logger.error(f"Error applying middleware '{name}': {e}", exc_info=True)
        
        logger.info(f"Applied {len(selected)} selected middleware to app")
    
    def get_middleware(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get middleware configuration by name.
        
        Args:
            name: Middleware name
        
        Returns:
            Middleware configuration or None
        """
        return self._middleware.get(name)
    
    def list_middleware(self) -> List[str]:
        """
        List all registered middleware names.
        
        Returns:
            List of middleware names
        """
        return list(self._middleware.keys())


# Singleton instance
_middleware_registry = MiddlewareRegistry()


def get_middleware_registry() -> MiddlewareRegistry:
    """Returns the singleton instance of the MiddlewareRegistry."""
    return _middleware_registry

