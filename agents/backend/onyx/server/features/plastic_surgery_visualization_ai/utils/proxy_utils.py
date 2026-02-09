"""Proxy pattern utilities."""

from typing import Any, Callable, Optional
import functools


class Proxy:
    """Generic proxy."""
    
    def __init__(self, target: Any):
        self._target = target
    
    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access."""
        return getattr(self._target, name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Proxy attribute setting."""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            setattr(self._target, name, value)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Proxy call."""
        return self._target(*args, **kwargs)


class LazyProxy:
    """Lazy loading proxy."""
    
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._target: Optional[Any] = None
    
    def _get_target(self) -> Any:
        """Get target, creating if necessary."""
        if self._target is None:
            self._target = self._factory()
        return self._target
    
    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access."""
        return getattr(self._get_target(), name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Proxy attribute setting."""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            setattr(self._get_target(), name, value)


class VirtualProxy:
    """Virtual proxy with access control."""
    
    def __init__(self, target: Any, access_check: Optional[Callable] = None):
        self._target = target
        self._access_check = access_check
    
    def _check_access(self) -> None:
        """Check access permission."""
        if self._access_check and not self._access_check():
            raise PermissionError("Access denied")
    
    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access."""
        self._check_access()
        return getattr(self._target, name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Proxy attribute setting."""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._check_access()
            setattr(self._target, name, value)


def proxy_method(func: Callable) -> Callable:
    """Decorator to proxy method calls."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, '_proxy_target'):
            return func(self._proxy_target, *args, **kwargs)
        return func(self, *args, **kwargs)
    return wrapper



