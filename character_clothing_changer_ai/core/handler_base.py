"""
Handler Base
============

Base classes for all handler types.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class HandlerConfig:
    """Handler configuration."""
    name: str
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandlerResult:
    """Handler execution result."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class BaseHandler(ABC):
    """Base handler interface."""
    
    def __init__(self, config: HandlerConfig):
        """
        Initialize handler.
        
        Args:
            config: Handler configuration
        """
        self.config = config
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_duration": 0.0
        }
    
    @abstractmethod
    async def handle(self, *args, **kwargs) -> HandlerResult:
        """
        Handle request.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Handler result
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        total = self.stats["total_calls"]
        success_rate = (
            self.stats["successful_calls"] / total
            if total > 0 else 0.0
        )
        avg_duration = (
            self.stats["total_duration"] / total
            if total > 0 else 0.0
        )
        
        return {
            "name": self.config.name,
            "enabled": self.config.enabled,
            "total_calls": total,
            "successful_calls": self.stats["successful_calls"],
            "failed_calls": self.stats["failed_calls"],
            "success_rate": success_rate,
            "avg_duration": avg_duration
        }
    
    def _update_stats(self, result: HandlerResult):
        """Update handler statistics."""
        self.stats["total_calls"] += 1
        if result.success:
            self.stats["successful_calls"] += 1
        else:
            self.stats["failed_calls"] += 1
        self.stats["total_duration"] += result.duration


class AsyncHandler(BaseHandler):
    """Async handler implementation."""
    
    def __init__(self, config: HandlerConfig, handler_func: Callable[..., Awaitable[Any]]):
        """
        Initialize async handler.
        
        Args:
            config: Handler configuration
            handler_func: Handler function
        """
        super().__init__(config)
        self.handler_func = handler_func
    
    async def handle(self, *args, **kwargs) -> HandlerResult:
        """Handle request with handler function."""
        if not self.config.enabled:
            return HandlerResult(
                success=False,
                error="Handler is disabled"
            )
        
        start = datetime.now()
        
        try:
            if self.config.timeout:
                result_data = await asyncio.wait_for(
                    self.handler_func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result_data = await self.handler_func(*args, **kwargs)
            
            duration = (datetime.now() - start).total_seconds()
            result = HandlerResult(
                success=True,
                data=result_data,
                duration=duration
            )
            
            self._update_stats(result)
            return result
            
        except asyncio.TimeoutError:
            duration = (datetime.now() - start).total_seconds()
            result = HandlerResult(
                success=False,
                error=f"Handler timeout after {self.config.timeout}s",
                duration=duration
            )
            self._update_stats(result)
            return result
            
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            result = HandlerResult(
                success=False,
                error=str(e),
                duration=duration
            )
            self._update_stats(result)
            return result


class HandlerChain:
    """Chain of handlers for sequential processing."""
    
    def __init__(self, name: str = "HandlerChain"):
        """
        Initialize handler chain.
        
        Args:
            name: Chain name
        """
        self.name = name
        self.handlers: List[BaseHandler] = []
    
    def add(self, handler: BaseHandler):
        """
        Add handler to chain.
        
        Args:
            handler: Handler instance
        """
        self.handlers.append(handler)
        logger.debug(f"Added handler {handler.config.name} to chain {self.name}")
    
    async def execute(self, *args, **kwargs) -> HandlerResult:
        """
        Execute handler chain.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Final handler result
        """
        context = {"args": args, "kwargs": kwargs}
        last_result = None
        
        for handler in self.handlers:
            if not handler.config.enabled:
                continue
            
            try:
                result = await handler.handle(*args, **kwargs)
                last_result = result
                
                if not result.success:
                    logger.warning(f"Handler {handler.config.name} failed: {result.error}")
                    return result
                
                # Update context with result
                context[handler.config.name] = result.data
                
            except Exception as e:
                return HandlerResult(
                    success=False,
                    error=f"Handler {handler.config.name} error: {str(e)}"
                )
        
        return last_result or HandlerResult(success=False, error="No handlers executed")

