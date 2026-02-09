"""
Execution Context
================

Context management for request execution.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Context variables
execution_context: ContextVar[Dict[str, Any]] = ContextVar('execution_context', default={})
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_context: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


@dataclass
class ExecutionContext:
    """Execution context for requests."""
    request_id: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    
    def finish(self, success: bool = True, error: Optional[str] = None):
        """
        Finish execution context.
        
        Args:
            success: Whether execution was successful
            error: Optional error message
        """
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error = error


class ContextManager:
    """Manager for execution contexts."""
    
    @staticmethod
    def set_context(context: ExecutionContext):
        """
        Set execution context.
        
        Args:
            context: Execution context
        """
        execution_context.set({
            "request_id": context.request_id,
            "user_id": context.user_id,
            "metadata": context.metadata,
            "start_time": context.start_time.isoformat()
        })
        request_id_context.set(context.request_id)
        if context.user_id:
            user_id_context.set(context.user_id)
    
    @staticmethod
    def get_context() -> Optional[Dict[str, Any]]:
        """Get current execution context."""
        return execution_context.get()
    
    @staticmethod
    def get_request_id() -> Optional[str]:
        """Get current request ID."""
        return request_id_context.get()
    
    @staticmethod
    def get_user_id() -> Optional[str]:
        """Get current user ID."""
        return user_id_context.get()
    
    @staticmethod
    def clear_context():
        """Clear execution context."""
        execution_context.set({})
        request_id_context.set(None)
        user_id_context.set(None)
    
    @staticmethod
    def with_context(context: ExecutionContext):
        """
        Decorator to set execution context.
        
        Args:
            context: Execution context
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    ContextManager.set_context(context)
                    try:
                        result = await func(*args, **kwargs)
                        context.finish(success=True)
                        return result
                    except Exception as e:
                        context.finish(success=False, error=str(e))
                        raise
                    finally:
                        ContextManager.clear_context()
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    ContextManager.set_context(context)
                    try:
                        result = func(*args, **kwargs)
                        context.finish(success=True)
                        return result
                    except Exception as e:
                        context.finish(success=False, error=str(e))
                        raise
                    finally:
                        ContextManager.clear_context()
                return sync_wrapper
        return decorator




