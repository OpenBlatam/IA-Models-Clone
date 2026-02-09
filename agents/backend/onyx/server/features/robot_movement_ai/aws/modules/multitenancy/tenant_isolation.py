"""
Tenant Isolation
================

Tenant data isolation.
"""

import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class TenantIsolation:
    """Tenant data isolation."""
    
    def __init__(self):
        self._isolation_strategy = "database"  # database, schema, row_level
        self._tenant_context: Dict[str, str] = {}
    
    def set_isolation_strategy(self, strategy: str):
        """Set isolation strategy."""
        valid_strategies = ["database", "schema", "row_level"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
        
        self._isolation_strategy = strategy
        logger.info(f"Set isolation strategy: {strategy}")
    
    def set_tenant_context(self, request_id: str, tenant_id: str):
        """Set tenant context for request."""
        self._tenant_context[request_id] = tenant_id
    
    def get_tenant_context(self, request_id: str) -> Optional[str]:
        """Get tenant context for request."""
        return self._tenant_context.get(request_id)
    
    def clear_tenant_context(self, request_id: str):
        """Clear tenant context."""
        self._tenant_context.pop(request_id, None)
    
    def isolate_query(self, query: str, tenant_id: str) -> str:
        """Isolate query by tenant."""
        if self._isolation_strategy == "database":
            # In production, route to tenant-specific database
            return query
        
        elif self._isolation_strategy == "schema":
            # Add schema prefix
            return f"SET search_path TO tenant_{tenant_id}; {query}"
        
        elif self._isolation_strategy == "row_level":
            # Add tenant filter
            if "WHERE" in query.upper():
                return f"{query} AND tenant_id = '{tenant_id}'"
            else:
                return f"{query} WHERE tenant_id = '{tenant_id}'"
        
        return query
    
    def tenant_aware(self, func: Callable) -> Callable:
        """Decorator to make function tenant-aware."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract tenant_id from context
            tenant_id = kwargs.get("tenant_id")
            if not tenant_id:
                raise ValueError("tenant_id required")
            
            # Isolate operation
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tenant_id = kwargs.get("tenant_id")
            if not tenant_id:
                raise ValueError("tenant_id required")
            
            return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper















