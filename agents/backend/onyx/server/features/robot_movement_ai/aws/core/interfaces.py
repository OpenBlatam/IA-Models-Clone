"""
Core Interfaces and Abstractions
=================================

Defines interfaces for all pluggable components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, Request, Response


class MiddlewarePlugin(ABC):
    """Interface for middleware plugins."""
    
    @abstractmethod
    def setup(self, app: FastAPI, config: Dict[str, Any]) -> FastAPI:
        """Setup middleware on FastAPI app."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        """Check if plugin is enabled."""
        pass


class MonitoringPlugin(ABC):
    """Interface for monitoring plugins."""
    
    @abstractmethod
    def setup(self, app: FastAPI, config: Dict[str, Any]) -> FastAPI:
        """Setup monitoring on FastAPI app."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        """Check if plugin is enabled."""
        pass


class SecurityPlugin(ABC):
    """Interface for security plugins."""
    
    @abstractmethod
    def setup(self, app: FastAPI, config: Dict[str, Any]) -> FastAPI:
        """Setup security on FastAPI app."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        """Check if plugin is enabled."""
        pass


class MessagingPlugin(ABC):
    """Interface for messaging plugins."""
    
    @abstractmethod
    def publish(self, event_type: str, data: Dict[str, Any], key: Optional[str] = None) -> bool:
        """Publish event."""
        pass
    
    @abstractmethod
    def subscribe(self, topic: str, handler: callable) -> bool:
        """Subscribe to topic."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        """Check if plugin is enabled."""
        pass


class CachePlugin(ABC):
    """Interface for cache plugins."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        """Check if plugin is enabled."""
        pass


class WorkerPlugin(ABC):
    """Interface for worker plugins."""
    
    @abstractmethod
    def enqueue_task(self, task_name: str, *args, **kwargs) -> str:
        """Enqueue task."""
        pass
    
    @abstractmethod
    def get_task_result(self, task_id: str) -> Any:
        """Get task result."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        """Check if plugin is enabled."""
        pass


class ServiceRegistry(ABC):
    """Interface for service registry."""
    
    @abstractmethod
    def register(self, name: str, service: Any) -> None:
        """Register a service."""
        pass
    
    @abstractmethod
    def get(self, name: str) -> Optional[Any]:
        """Get a service by name."""
        pass
    
    @abstractmethod
    def unregister(self, name: str) -> None:
        """Unregister a service."""
        pass
    
    @abstractmethod
    def list_services(self) -> List[str]:
        """List all registered services."""
        pass















