"""
Frontier Model Polyglot — System 5.0 Registry.
Global component discovery and dependency resolution.
"""
from typing import Dict, Any, Type, TypeVar, Optional

T = TypeVar("T")

class Registry:
    """Centralized service registry for the polyglot training system."""
    def __init__(self):
        self._services: Dict[str, Any] = {}

    def register(self, name: str, service: Any):
        self._services[name] = service

    def get(self, name: str) -> Optional[Any]:
        return self._services.get(name)

    def resolve(self, service_type: Type[T]) -> Optional[T]:
        """Resolve a service by its type."""
        for service in self._services.values():
            if isinstance(service, service_type):
                return service
        return None

registry = Registry()
