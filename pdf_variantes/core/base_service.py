"""
PDF Variantes - Base Service Class
Abstract base class for all services with common functionality
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

from ..utils.config import Settings


class BaseService(ABC):
    """Base service class with common functionality"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the service - override if needed"""
        if not self._initialized:
            await self._initialize()
            self._initialized = True
            self.logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Service-specific initialization - must be implemented"""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup resources - override if needed"""
        if self._initialized:
            await self._cleanup()
            self._initialized = False
            self.logger.info(f"{self.__class__.__name__} cleaned up")
    
    async def _cleanup(self) -> None:
        """Service-specific cleanup - override if needed"""
        pass
    
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self._initialized
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status for health checks"""
        return {
            "name": self.__class__.__name__,
            "initialized": self._initialized,
            "status": "ready" if self._initialized else "not_initialized"
        }


class AsyncServiceMixin:
    """Mixin for async service operations"""
    
    async def health_check(self) -> Dict[str, Any]:
        """Default health check - override if needed"""
        return {"status": "healthy"}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics - override if needed"""
        return {}


class EventEmitterMixin:
    """Mixin for event-driven services"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._event_handlers: Dict[str, list] = {}
    
    def on(self, event: str, handler: callable):
        """Register event handler"""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    def off(self, event: str, handler: callable):
        """Unregister event handler"""
        if event in self._event_handlers:
            try:
                self._event_handlers[event].remove(handler)
            except ValueError:
                pass
    
    async def emit(self, event: str, *args, **kwargs):
        """Emit event to registered handlers"""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                if hasattr(handler, '__call__'):
                    if hasattr(handler, '__await__'):
                        await handler(*args, **kwargs)
                    else:
                        handler(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event}: {e}")






