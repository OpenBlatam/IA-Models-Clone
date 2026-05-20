"""
Promise utilities
Promise-like patterns for async operations
"""

from typing import TypeVar, Callable, Optional, Any
from asyncio import Future
import asyncio

T = TypeVar('T')
U = TypeVar('U')


class Promise:
    """
    Promise-like class for async operations
    """
    
    def __init__(self, executor: Optional[Callable] = None):
        self._future: Future = Future()
        self._resolved = False
        self._rejected = False
        self._value: Optional[T] = None
        self._error: Optional[Exception] = None
        
        if executor:
            asyncio.create_task(self._execute(executor))
    
    async def _execute(self, executor: Callable):
        """Execute promise executor"""
        try:
            result = await executor() if asyncio.iscoroutinefunction(executor) else executor()
            await self.resolve(result)
        except Exception as e:
            await self.reject(e)
    
    async def resolve(self, value: T) -> 'Promise':
        """Resolve promise with value"""
        if self._resolved or self._rejected:
            return self
        
        self._resolved = True
        self._value = value
        self._future.set_result(value)
        return self
    
    async def reject(self, error: Exception) -> 'Promise':
        """Reject promise with error"""
        if self._resolved or self._rejected:
            return self
        
        self._rejected = True
        self._error = error
        self._future.set_exception(error)
        return self
    
    def then(self, on_fulfilled: Callable[[T], U], on_rejected: Optional[Callable] = None) -> 'Promise':
        """Chain promise with then"""
        async def executor():
            try:
                value = await self._future
                result = on_fulfilled(value)
                return await result if isinstance(result, Promise) else result
            except Exception as e:
                if on_rejected:
                    return on_rejected(e)
                raise
        
        return Promise(executor)
    
    def catch(self, on_rejected: Callable) -> 'Promise':
        """Chain promise with catch"""
        return self.then(lambda x: x, on_rejected)
    
    async def await_value(self) -> T:
        """Await promise value"""
        return await self._future
    
    @classmethod
    def resolve_value(cls, value: T) -> 'Promise':
        """Create resolved promise"""
        promise = cls()
        asyncio.create_task(promise.resolve(value))
        return promise
    
    @classmethod
    def reject_value(cls, error: Exception) -> 'Promise':
        """Create rejected promise"""
        promise = cls()
        asyncio.create_task(promise.reject(error))
        return promise


def create_promise(executor: Callable) -> Promise:
    """Create new promise"""
    return Promise(executor)

