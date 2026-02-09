"""Chain of Responsibility pattern utilities and method chaining."""

from typing import Any, Optional, Callable, List, TypeVar
from abc import ABC, abstractmethod
from functools import reduce

T = TypeVar('T')


class Handler(ABC):
    """Base handler interface."""
    
    def __init__(self):
        self._next: Optional['Handler'] = None
    
    def set_next(self, handler: 'Handler') -> 'Handler':
        """Set next handler in chain."""
        self._next = handler
        return handler
    
    @abstractmethod
    def handle(self, request: Any) -> Any:
        """Handle request."""
        pass
    
    def _handle_next(self, request: Any) -> Any:
        """Pass to next handler."""
        if self._next:
            return self._next.handle(request)
        return None


class BaseHandler(Handler):
    """Base handler implementation."""
    
    def __init__(self, name: str = ""):
        super().__init__()
        self.name = name
    
    def handle(self, request: Any) -> Any:
        """Handle request."""
        if self.can_handle(request):
            return self.process(request)
        return self._handle_next(request)
    
    def can_handle(self, request: Any) -> bool:
        """Check if handler can process request."""
        return True
    
    def process(self, request: Any) -> Any:
        """Process request."""
        return request


class FunctionHandler(Handler):
    """Handler from function."""
    
    def __init__(self, handler_func: Callable, condition: Optional[Callable] = None):
        super().__init__()
        self.handler_func = handler_func
        self.condition = condition
    
    def handle(self, request: Any) -> Any:
        """Handle request."""
        if self.condition is None or self.condition(request):
            return self.handler_func(request)
        return self._handle_next(request)


class ChainBuilder:
    """Builder for handler chains."""
    
    def __init__(self):
        self.handlers: List[Handler] = []
        self.first: Optional[Handler] = None
    
    def add(self, handler: Handler) -> 'ChainBuilder':
        """Add handler to chain."""
        self.handlers.append(handler)
        return self
    
    def build(self) -> Optional[Handler]:
        """Build chain."""
        if not self.handlers:
            return None
        
        self.first = self.handlers[0]
        current = self.first
        
        for handler in self.handlers[1:]:
            current.set_next(handler)
            current = handler
        
        return self.first
    
    def execute(self, request: Any) -> Any:
        """Execute chain with request."""
        if not self.first:
            self.build()
        
        if self.first:
            return self.first.handle(request)
        return None


class ConditionalHandler(BaseHandler):
    """Handler with condition."""
    
    def __init__(self, condition: Callable, name: str = ""):
        super().__init__(name)
        self.condition = condition
    
    def can_handle(self, request: Any) -> bool:
        """Check condition."""
        return self.condition(request)


class AsyncHandler(Handler):
    """Async handler interface."""
    
    @abstractmethod
    async def handle_async(self, request: Any) -> Any:
        """Handle request asynchronously."""
        pass
    
    def handle(self, request: Any) -> Any:
        """Sync handle (raises error)."""
        raise NotImplementedError("Use handle_async() for async handlers")
    
    async def _handle_next_async(self, request: Any) -> Any:
        """Pass to next handler asynchronously."""
        if self._next:
            if isinstance(self._next, AsyncHandler):
                return await self._next.handle_async(request)
            return self._next.handle(request)
        return None


class MiddlewareHandler(BaseHandler):
    """Handler that can modify request/response."""
    
    def handle(self, request: Any) -> Any:
        """Handle with middleware processing."""
        modified_request = self.before_handle(request)
        result = self._handle_next(modified_request)
        return self.after_handle(result, request)
    
    def before_handle(self, request: Any) -> Any:
        """Process before passing to next handler."""
        return request
    
    def after_handle(self, result: Any, original_request: Any) -> Any:
        """Process after receiving result."""
        return result


class ErrorHandler(BaseHandler):
    """Handler for error processing."""
    
    def __init__(self, error_type: type = Exception, name: str = ""):
        super().__init__(name)
        self.error_type = error_type
    
    def handle(self, request: Any) -> Any:
        """Handle with error catching."""
        try:
            return self._handle_next(request)
        except self.error_type as e:
            return self.handle_error(e, request)
    
    def handle_error(self, error: Exception, request: Any) -> Any:
        """Handle error."""
        raise error


def create_chain(*handlers: Handler) -> Optional[Handler]:
    """Create handler chain."""
    builder = ChainBuilder()
    for handler in handlers:
        builder.add(handler)
    return builder.build()


# Method chaining utilities from chain_utils.py
class Chain:
    """Method chaining utility."""
    
    def __init__(self, value: Any):
        self._value = value
    
    def map(self, func: Callable) -> 'Chain':
        """Map over value."""
        if isinstance(self._value, list):
            self._value = [func(item) for item in self._value]
        else:
            self._value = func(self._value)
        return self
    
    def filter(self, predicate: Callable) -> 'Chain':
        """Filter value."""
        if isinstance(self._value, list):
            self._value = [item for item in self._value if predicate(item)]
        return self
    
    def reduce(self, func: Callable, initial: Any = None) -> 'Chain':
        """Reduce value."""
        if isinstance(self._value, list):
            if initial is not None:
                self._value = reduce(func, self._value, initial)
            else:
                self._value = reduce(func, self._value)
        return self
    
    def tap(self, func: Callable) -> 'Chain':
        """Tap into value (side effect)."""
        func(self._value)
        return self
    
    def value(self) -> Any:
        """Get chained value."""
        return self._value


def chain(value: Any) -> Chain:
    """Create chain from value."""
    return Chain(value)

