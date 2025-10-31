"""
PDF Variantes API - Architecture Layers
Clean Architecture implementation with clear layer separation
"""

from abc import ABC, abstractmethod
from typing import Protocol, Any, Optional, List
from datetime import datetime


# ============================================================================
# Domain Layer - Business Logic and Entities
# ============================================================================

class DomainEntity(ABC):
    """Base domain entity"""
    pass


class DomainService(ABC):
    """Base domain service - business logic"""
    pass


# ============================================================================
# Application Layer - Use Cases and Application Services
# ============================================================================

class UseCase(ABC):
    """Base use case following CQRS pattern"""
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the use case"""
        pass


class Command(Protocol):
    """Command for write operations"""
    pass


class Query(Protocol):
    """Query for read operations"""
    pass


class CommandHandler(ABC):
    """Handle commands (write operations)"""
    
    @abstractmethod
    async def handle(self, command: Command) -> Any:
        """Handle command"""
        pass


class QueryHandler(ABC):
    """Handle queries (read operations)"""
    
    @abstractmethod
    async def handle(self, query: Query) -> Any:
        """Handle query"""
        pass


# ============================================================================
# Infrastructure Layer - External Dependencies
# ============================================================================

class Repository(ABC):
    """Base repository interface"""
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[Any]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def save(self, entity: Any) -> Any:
        """Save entity"""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity"""
        pass


class CacheRepository(ABC):
    """Cache repository interface"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete from cache"""
        pass


class EventBus(ABC):
    """Event bus for domain events"""
    
    @abstractmethod
    async def publish(self, event: Any) -> bool:
        """Publish event"""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: str, handler: callable) -> bool:
        """Subscribe to event"""
        pass


# ============================================================================
# Presentation Layer - API and Controllers
# ============================================================================

class Controller(ABC):
    """Base controller"""
    
    @abstractmethod
    async def handle(self, request: Any) -> Any:
        """Handle request"""
        pass


class Presenter(ABC):
    """Base presenter for formatting responses"""
    
    @abstractmethod
    def present(self, data: Any) -> dict:
        """Format data for response"""
        pass


# ============================================================================
# Dependency Injection Container
# ============================================================================

class Container:
    """Dependency injection container"""
    
    def __init__(self):
        self._services: dict = {}
        self._singletons: dict = {}
    
    def register(self, interface: type, implementation: type, singleton: bool = False):
        """Register service"""
        self._services[interface] = {
            "implementation": implementation,
            "singleton": singleton
        }
    
    def resolve(self, interface: type) -> Any:
        """Resolve service"""
        if interface in self._singletons:
            return self._singletons[interface]
        
        service_info = self._services.get(interface)
        if not service_info:
            raise ValueError(f"Service {interface} not registered")
        
        implementation = service_info["implementation"]
        instance = implementation()
        
        if service_info["singleton"]:
            self._singletons[interface] = instance
        
        return instance
    
    def register_instance(self, interface: type, instance: Any):
        """Register existing instance"""
        self._singletons[interface] = instance






