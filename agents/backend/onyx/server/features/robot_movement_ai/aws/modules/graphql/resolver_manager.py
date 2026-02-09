"""
Resolver Manager
================

GraphQL resolver management.
"""

import logging
from typing import Dict, Any, Optional, Callable
import asyncio

logger = logging.getLogger(__name__)


class ResolverManager:
    """GraphQL resolver manager."""
    
    def __init__(self):
        self._resolvers: Dict[str, Callable] = {}
        self._field_resolvers: Dict[str, Dict[str, Callable]] = {}  # type -> field -> resolver
        self._middleware: List[Callable] = []
    
    def register_resolver(self, name: str, resolver: Callable):
        """Register resolver."""
        self._resolvers[name] = resolver
        logger.info(f"Registered resolver: {name}")
    
    def register_field_resolver(self, type_name: str, field_name: str, resolver: Callable):
        """Register field resolver."""
        if type_name not in self._field_resolvers:
            self._field_resolvers[type_name] = {}
        
        self._field_resolvers[type_name][field_name] = resolver
        logger.info(f"Registered field resolver: {type_name}.{field_name}")
    
    def add_middleware(self, middleware: Callable):
        """Add resolver middleware."""
        self._middleware.append(middleware)
        logger.info("Added resolver middleware")
    
    async def resolve(self, resolver_name: str, *args, **kwargs) -> Any:
        """Resolve query/mutation."""
        if resolver_name not in self._resolvers:
            raise ValueError(f"Resolver {resolver_name} not found")
        
        resolver = self._resolvers[resolver_name]
        
        # Apply middleware
        for middleware in self._middleware:
            resolver = middleware(resolver)
        
        # Execute resolver
        if asyncio.iscoroutinefunction(resolver):
            return await resolver(*args, **kwargs)
        else:
            return resolver(*args, **kwargs)
    
    async def resolve_field(self, type_name: str, field_name: str, parent: Any, *args, **kwargs) -> Any:
        """Resolve field."""
        if type_name in self._field_resolvers:
            if field_name in self._field_resolvers[type_name]:
                resolver = self._field_resolvers[type_name][field_name]
                
                if asyncio.iscoroutinefunction(resolver):
                    return await resolver(parent, *args, **kwargs)
                else:
                    return resolver(parent, *args, **kwargs)
        
        # Default: return field from parent
        if hasattr(parent, field_name):
            return getattr(parent, field_name)
        
        if isinstance(parent, dict):
            return parent.get(field_name)
        
        return None
    
    def get_resolver(self, name: str) -> Optional[Callable]:
        """Get resolver by name."""
        return self._resolvers.get(name)
    
    def get_resolver_stats(self) -> Dict[str, Any]:
        """Get resolver statistics."""
        return {
            "total_resolvers": len(self._resolvers),
            "total_field_resolvers": sum(
                len(fields) for fields in self._field_resolvers.values()
            ),
            "middleware_count": len(self._middleware)
        }















