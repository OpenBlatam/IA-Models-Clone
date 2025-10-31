"""
Base Handler Implementation

Ultra-specialized base handler with advanced features for
event processing, request handling, and response management.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import weakref
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class HandlerType(Enum):
    """Handler type enumeration"""
    EVENT = "event"
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    ROUTING = "routing"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    LOGGING = "logging"
    MONITORING = "monitoring"


class HandlerPriority(Enum):
    """Handler priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class HandlerConfig:
    """Handler configuration"""
    name: str
    handler_type: HandlerType
    priority: HandlerPriority = HandlerPriority.NORMAL
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    batch_size: int = 1
    max_concurrent: int = 10
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandlerContext:
    """Handler execution context"""
    handler_name: str
    event_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    data: Any = None
    result: Any = None
    exception: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseHandler(ABC, Generic[T, R]):
    """Base handler with advanced features"""
    
    def __init__(self, config: HandlerConfig):
        self.config = config
        self._enabled = config.enabled
        self._event_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._callbacks: List[Callable] = []
        self._error_handlers: List[Callable] = []
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
    
    @abstractmethod
    async def _handle(self, data: T, context: HandlerContext) -> R:
        """Handle data (override in subclasses)"""
        pass
    
    async def handle(self, data: T, event_type: str = "default") -> R:
        """Handle data with context"""
        if not self._enabled:
            raise RuntimeError(f"Handler '{self.config.name}' is disabled")
        
        context = HandlerContext(
            handler_name=self.config.name,
            event_type=event_type,
            start_time=datetime.utcnow(),
            data=data
        )
        
        async with self._semaphore:
            try:
                # Pre-handle hook
                await self._pre_handle(context)
                
                # Handle data
                result = await asyncio.wait_for(
                    self._handle(data, context),
                    timeout=self.config.timeout
                )
                
                context.result = result
                
                # Post-handle hook
                await self._post_handle(context)
                
                return result
                
            except Exception as e:
                context.exception = e
                await self._handle_error(context)
                raise
    
    async def handle_batch(self, data_list: List[T], event_type: str = "default") -> List[R]:
        """Handle multiple data items"""
        if not self._enabled:
            raise RuntimeError(f"Handler '{self.config.name}' is disabled")
        
        # Process in batches
        results = []
        for i in range(0, len(data_list), self.config.batch_size):
            batch = data_list[i:i + self.config.batch_size]
            
            # Process batch concurrently
            tasks = [
                self.handle(data, event_type)
                for data in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    async def _pre_handle(self, context: HandlerContext) -> None:
        """Pre-handle hook (override in subclasses)"""
        pass
    
    async def _post_handle(self, context: HandlerContext) -> None:
        """Post-handle hook (override in subclasses)"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        # Update metrics
        self._update_metrics(context)
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context)
                else:
                    callback(context)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    async def _handle_error(self, context: HandlerContext) -> None:
        """Handle errors"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        # Update metrics
        self._update_metrics(context)
        
        # Call error handlers
        for handler in self._error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(context)
                else:
                    handler(context)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    def _update_metrics(self, context: HandlerContext) -> None:
        """Update handler metrics"""
        self._event_count += 1
        if context.exception is None:
            self._success_count += 1
        else:
            self._error_count += 1
        
        if context.duration:
            self._total_duration += context.duration
    
    def add_callback(self, callback: Callable) -> None:
        """Add callback for events"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def add_error_handler(self, handler: Callable) -> None:
        """Add error handler"""
        self._error_handlers.append(handler)
    
    def remove_error_handler(self, handler: Callable) -> None:
        """Remove error handler"""
        if handler in self._error_handlers:
            self._error_handlers.remove(handler)
    
    def enable(self) -> None:
        """Enable handler"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable handler"""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if handler is enabled"""
        return self._enabled
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get handler metrics"""
        avg_duration = (
            self._total_duration / self._event_count
            if self._event_count > 0 else 0
        )
        
        success_rate = (
            self._success_count / self._event_count
            if self._event_count > 0 else 0
        )
        
        return {
            "name": self.config.name,
            "type": self.config.handler_type.value,
            "priority": self.config.priority.value,
            "enabled": self._enabled,
            "event_count": self._event_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": success_rate,
            "total_duration": self._total_duration,
            "average_duration": avg_duration,
            "max_concurrent": self.config.max_concurrent,
            "current_concurrent": self.config.max_concurrent - self._semaphore._value
        }
    
    def reset_metrics(self) -> None:
        """Reset handler metrics"""
        self._event_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', enabled={self._enabled})"


class HandlerChain:
    """Chain of handlers with priority ordering"""
    
    def __init__(self):
        self._handlers: List[BaseHandler] = []
        self._lock = asyncio.Lock()
    
    def add_handler(self, handler: BaseHandler) -> None:
        """Add handler to chain"""
        self._handlers.append(handler)
        # Sort by priority (higher priority first)
        self._handlers.sort(
            key=lambda h: h.config.priority.value,
            reverse=True
        )
    
    def remove_handler(self, name: str) -> None:
        """Remove handler from chain"""
        self._handlers = [
            h for h in self._handlers
            if h.config.name != name
        ]
    
    async def handle(self, data: Any, event_type: str = "default") -> Any:
        """Handle data through handler chain"""
        result = data
        
        for handler in self._handlers:
            if handler.is_enabled():
                try:
                    result = await handler.handle(result, event_type)
                except Exception as e:
                    logger.error(f"Error in handler '{handler.config.name}': {e}")
                    # Continue to next handler or re-raise based on configuration
                    raise
        
        return result
    
    async def handle_batch(self, data_list: List[Any], event_type: str = "default") -> List[Any]:
        """Handle multiple data items through handler chain"""
        results = data_list
        
        for handler in self._handlers:
            if handler.is_enabled():
                try:
                    results = await handler.handle_batch(results, event_type)
                except Exception as e:
                    logger.error(f"Error in handler '{handler.config.name}': {e}")
                    raise
        
        return results
    
    def get_handlers(self) -> List[BaseHandler]:
        """Get all handlers"""
        return self._handlers.copy()
    
    def get_handlers_by_type(self, handler_type: HandlerType) -> List[BaseHandler]:
        """Get handlers by type"""
        return [
            h for h in self._handlers
            if h.config.handler_type == handler_type
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all handlers"""
        return {
            handler.config.name: handler.get_metrics()
            for handler in self._handlers
        }


class HandlerRegistry:
    """Registry for managing handlers"""
    
    def __init__(self):
        self._handlers: Dict[str, BaseHandler] = {}
        self._chains: Dict[str, HandlerChain] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, handler: BaseHandler) -> None:
        """Register handler"""
        async with self._lock:
            self._handlers[handler.config.name] = handler
            logger.info(f"Registered handler: {handler.config.name}")
    
    async def unregister(self, name: str) -> None:
        """Unregister handler"""
        async with self._lock:
            if name in self._handlers:
                del self._handlers[name]
                logger.info(f"Unregistered handler: {name}")
    
    def get(self, name: str) -> Optional[BaseHandler]:
        """Get handler by name"""
        return self._handlers.get(name)
    
    def get_by_type(self, handler_type: HandlerType) -> List[BaseHandler]:
        """Get handlers by type"""
        return [
            handler for handler in self._handlers.values()
            if handler.config.handler_type == handler_type
        ]
    
    def create_chain(self, name: str, handler_names: List[str]) -> HandlerChain:
        """Create handler chain"""
        chain = HandlerChain()
        
        for handler_name in handler_names:
            handler = self.get(handler_name)
            if handler:
                chain.add_handler(handler)
        
        self._chains[name] = chain
        return chain
    
    def get_chain(self, name: str) -> Optional[HandlerChain]:
        """Get handler chain"""
        return self._chains.get(name)
    
    def list_all(self) -> List[BaseHandler]:
        """List all handlers"""
        return list(self._handlers.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all handlers"""
        return {
            name: handler.get_metrics()
            for name, handler in self._handlers.items()
        }


# Global handler registry
handler_registry = HandlerRegistry()


# Convenience functions
async def register_handler(handler: BaseHandler):
    """Register handler"""
    await handler_registry.register(handler)


def get_handler(name: str) -> Optional[BaseHandler]:
    """Get handler by name"""
    return handler_registry.get(name)


def create_handler_chain(name: str, handler_names: List[str]) -> HandlerChain:
    """Create handler chain"""
    return handler_registry.create_chain(name, handler_names)


# Handler factory functions
def create_handler(handler_type: HandlerType, name: str, **kwargs) -> BaseHandler:
    """Create handler by type"""
    config = HandlerConfig(
        name=name,
        handler_type=handler_type,
        **kwargs
    )
    
    # This would be implemented with specific handler classes
    # For now, return a placeholder
    raise NotImplementedError(f"Handler type {handler_type} not implemented yet")


# Common handler combinations
def request_processing_chain(name: str = "request_processing") -> HandlerChain:
    """Create request processing handler chain"""
    return create_handler_chain(name, [
        "validation",
        "authentication",
        "authorization",
        "transformation",
        "routing"
    ])


def response_processing_chain(name: str = "response_processing") -> HandlerChain:
    """Create response processing handler chain"""
    return create_handler_chain(name, [
        "transformation",
        "formatting",
        "logging",
        "monitoring"
    ])


def error_processing_chain(name: str = "error_processing") -> HandlerChain:
    """Create error processing handler chain"""
    return create_handler_chain(name, [
        "error_handling",
        "logging",
        "monitoring",
        "fallback"
    ])


def event_processing_chain(name: str = "event_processing") -> HandlerChain:
    """Create event processing handler chain"""
    return create_handler_chain(name, [
        "validation",
        "transformation",
        "routing",
        "logging",
        "monitoring"
    ])





















