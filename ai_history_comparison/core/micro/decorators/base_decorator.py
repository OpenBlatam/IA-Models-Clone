"""
Base Decorator Implementation

Ultra-specialized base decorator with advanced features for
cross-cutting concerns and aspect-oriented programming.
"""

import asyncio
import logging
import functools
import time
import inspect
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


class DecoratorType(Enum):
    """Decorator type enumeration"""
    PERFORMANCE = "performance"
    CACHING = "caching"
    RETRY = "retry"
    VALIDATION = "validation"
    LOGGING = "logging"
    SECURITY = "security"
    MONITORING = "monitoring"
    RATE_LIMITING = "rate_limiting"
    ASYNC = "async"
    ERROR_HANDLING = "error_handling"


class DecoratorPriority(Enum):
    """Decorator priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DecoratorConfig:
    """Decorator configuration"""
    name: str
    decorator_type: DecoratorType
    priority: DecoratorPriority = DecoratorPriority.NORMAL
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    cache_ttl: Optional[float] = None
    rate_limit: Optional[int] = None
    rate_window: Optional[float] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Execution context for decorators"""
    function_name: str
    module_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    result: Any = None
    exception: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDecorator(ABC):
    """Base decorator with advanced features"""
    
    def __init__(self, config: DecoratorConfig):
        self.config = config
        self._enabled = config.enabled
        self._execution_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._callbacks: List[Callable] = []
        self._error_handlers: List[Callable] = []
        self._lock = asyncio.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Apply decorator to function"""
        if not self._enabled:
            return func
        
        if asyncio.iscoroutinefunction(func):
            return self._wrap_async_function(func)
        else:
            return self._wrap_sync_function(func)
    
    def _wrap_sync_function(self, func: Callable) -> Callable:
        """Wrap synchronous function"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = ExecutionContext(
                function_name=func.__name__,
                module_name=func.__module__,
                start_time=datetime.utcnow(),
                args=args,
                kwargs=kwargs
            )
            
            try:
                # Pre-execution hook
                self._pre_execute(context)
                
                # Execute function
                result = func(*args, **kwargs)
                context.result = result
                
                # Post-execution hook
                self._post_execute(context)
                
                return result
                
            except Exception as e:
                context.exception = e
                self._handle_error(context)
                raise
        
        return wrapper
    
    def _wrap_async_function(self, func: Callable) -> Callable:
        """Wrap asynchronous function"""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            context = ExecutionContext(
                function_name=func.__name__,
                module_name=func.__module__,
                start_time=datetime.utcnow(),
                args=args,
                kwargs=kwargs
            )
            
            try:
                # Pre-execution hook
                await self._pre_execute_async(context)
                
                # Execute function
                result = await func(*args, **kwargs)
                context.result = result
                
                # Post-execution hook
                await self._post_execute_async(context)
                
                return result
                
            except Exception as e:
                context.exception = e
                await self._handle_error_async(context)
                raise
        
        return wrapper
    
    @abstractmethod
    def _pre_execute(self, context: ExecutionContext) -> None:
        """Pre-execution hook (override in subclasses)"""
        pass
    
    @abstractmethod
    def _post_execute(self, context: ExecutionContext) -> None:
        """Post-execution hook (override in subclasses)"""
        pass
    
    async def _pre_execute_async(self, context: ExecutionContext) -> None:
        """Pre-execution hook for async functions"""
        self._pre_execute(context)
    
    async def _post_execute_async(self, context: ExecutionContext) -> None:
        """Post-execution hook for async functions"""
        self._post_execute(context)
    
    def _handle_error(self, context: ExecutionContext) -> None:
        """Handle execution errors"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        self._error_count += 1
        self._total_duration += context.duration
        
        # Call error handlers
        for handler in self._error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(context))
                else:
                    handler(context)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    async def _handle_error_async(self, context: ExecutionContext) -> None:
        """Handle execution errors for async functions"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        self._error_count += 1
        self._total_duration += context.duration
        
        # Call error handlers
        for handler in self._error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(context)
                else:
                    handler(context)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    def _update_metrics(self, context: ExecutionContext) -> None:
        """Update execution metrics"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        self._execution_count += 1
        if context.exception is None:
            self._success_count += 1
        else:
            self._error_count += 1
        
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
        """Enable decorator"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable decorator"""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if decorator is enabled"""
        return self._enabled
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get decorator metrics"""
        avg_duration = (
            self._total_duration / self._execution_count
            if self._execution_count > 0 else 0
        )
        
        success_rate = (
            self._success_count / self._execution_count
            if self._execution_count > 0 else 0
        )
        
        return {
            "name": self.config.name,
            "type": self.config.decorator_type.value,
            "priority": self.config.priority.value,
            "enabled": self._enabled,
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": success_rate,
            "total_duration": self._total_duration,
            "average_duration": avg_duration
        }
    
    def reset_metrics(self) -> None:
        """Reset decorator metrics"""
        self._execution_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_duration = 0.0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', enabled={self._enabled})"


class DecoratorChain:
    """Chain of decorators with priority ordering"""
    
    def __init__(self):
        self._decorators: List[BaseDecorator] = []
        self._lock = asyncio.Lock()
    
    def add_decorator(self, decorator: BaseDecorator) -> None:
        """Add decorator to chain"""
        self._decorators.append(decorator)
        # Sort by priority (higher priority first)
        self._decorators.sort(
            key=lambda d: d.config.priority.value,
            reverse=True
        )
    
    def remove_decorator(self, name: str) -> None:
        """Remove decorator from chain"""
        self._decorators = [
            d for d in self._decorators
            if d.config.name != name
        ]
    
    def apply_to_function(self, func: Callable) -> Callable:
        """Apply all decorators to function"""
        result = func
        
        # Apply decorators in priority order
        for decorator in self._decorators:
            if decorator.is_enabled():
                result = decorator(result)
        
        return result
    
    def get_decorators(self) -> List[BaseDecorator]:
        """Get all decorators"""
        return self._decorators.copy()
    
    def get_decorators_by_type(self, decorator_type: DecoratorType) -> List[BaseDecorator]:
        """Get decorators by type"""
        return [
            d for d in self._decorators
            if d.config.decorator_type == decorator_type
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all decorators"""
        return {
            decorator.config.name: decorator.get_metrics()
            for decorator in self._decorators
        }


class DecoratorRegistry:
    """Registry for managing decorators"""
    
    def __init__(self):
        self._decorators: Dict[str, BaseDecorator] = {}
        self._chains: Dict[str, DecoratorChain] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, decorator: BaseDecorator) -> None:
        """Register decorator"""
        async with self._lock:
            self._decorators[decorator.config.name] = decorator
            logger.info(f"Registered decorator: {decorator.config.name}")
    
    async def unregister(self, name: str) -> None:
        """Unregister decorator"""
        async with self._lock:
            if name in self._decorators:
                del self._decorators[name]
                logger.info(f"Unregistered decorator: {name}")
    
    def get(self, name: str) -> Optional[BaseDecorator]:
        """Get decorator by name"""
        return self._decorators.get(name)
    
    def get_by_type(self, decorator_type: DecoratorType) -> List[BaseDecorator]:
        """Get decorators by type"""
        return [
            decorator for decorator in self._decorators.values()
            if decorator.config.decorator_type == decorator_type
        ]
    
    def create_chain(self, name: str, decorator_names: List[str]) -> DecoratorChain:
        """Create decorator chain"""
        chain = DecoratorChain()
        
        for decorator_name in decorator_names:
            decorator = self.get(decorator_name)
            if decorator:
                chain.add_decorator(decorator)
        
        self._chains[name] = chain
        return chain
    
    def get_chain(self, name: str) -> Optional[DecoratorChain]:
        """Get decorator chain"""
        return self._chains.get(name)
    
    def list_all(self) -> List[BaseDecorator]:
        """List all decorators"""
        return list(self._decorators.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all decorators"""
        return {
            name: decorator.get_metrics()
            for name, decorator in self._decorators.items()
        }


# Global decorator registry
decorator_registry = DecoratorRegistry()


# Convenience functions
def register_decorator(decorator: BaseDecorator):
    """Register decorator"""
    asyncio.create_task(decorator_registry.register(decorator))


def get_decorator(name: str) -> Optional[BaseDecorator]:
    """Get decorator by name"""
    return decorator_registry.get(name)


def create_decorator_chain(name: str, decorator_names: List[str]) -> DecoratorChain:
    """Create decorator chain"""
    return decorator_registry.create_chain(name, decorator_names)


# Decorator factory functions
def create_decorator(decorator_type: DecoratorType, name: str, **kwargs) -> BaseDecorator:
    """Create decorator by type"""
    config = DecoratorConfig(
        name=name,
        decorator_type=decorator_type,
        **kwargs
    )
    
    # This would be implemented with specific decorator classes
    # For now, return a placeholder
    raise NotImplementedError(f"Decorator type {decorator_type} not implemented yet")


# Common decorator combinations
def performance_monitoring(name: str = "performance_monitoring") -> DecoratorChain:
    """Create performance monitoring decorator chain"""
    return create_decorator_chain(name, [
        "timing",
        "profiling",
        "metrics"
    ])


def error_handling_with_retry(name: str = "error_handling_retry") -> DecoratorChain:
    """Create error handling with retry decorator chain"""
    return create_decorator_chain(name, [
        "error_handling",
        "retry",
        "logging"
    ])


def security_with_validation(name: str = "security_validation") -> DecoratorChain:
    """Create security with validation decorator chain"""
    return create_decorator_chain(name, [
        "authentication",
        "authorization",
        "validation",
        "audit"
    ])





















