"""
Query Bus
=========

CQRS query bus.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Query:
    """Query definition."""
    type: str
    params: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class QueryBus:
    """Query bus for CQRS."""
    
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        self._query_history: list[Query] = []
    
    def register_handler(self, query_type: str, handler: Callable, cache_ttl: Optional[float] = None):
        """Register query handler."""
        self._handlers[query_type] = {
            "handler": handler,
            "cache_ttl": cache_ttl
        }
        logger.info(f"Registered query handler: {query_type}")
    
    async def dispatch(self, query: Query, use_cache: bool = True) -> Any:
        """Dispatch query."""
        if query.type not in self._handlers:
            raise ValueError(f"No handler registered for query: {query.type}")
        
        handler_info = self._handlers[query.type]
        handler = handler_info["handler"]
        cache_ttl = handler_info.get("cache_ttl")
        
        # Check cache
        cache_key = f"{query.type}:{str(query.params)}"
        if use_cache and cache_key in self._cache:
            if cache_ttl:
                if datetime.now() < self._cache_ttl.get(cache_key, datetime.min):
                    return self._cache[cache_key]
            else:
                return self._cache[cache_key]
        
        # Execute handler
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(query.params)
            else:
                result = await asyncio.to_thread(handler, query.params)
            
            # Cache result
            if use_cache and cache_ttl:
                self._cache[cache_key] = result
                self._cache_ttl[cache_key] = datetime.now() + asyncio.get_event_loop().time() + cache_ttl
            
            self._query_history.append(query)
            return result
        
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def clear_cache(self, query_type: Optional[str] = None):
        """Clear query cache."""
        if query_type:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{query_type}:")]
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._cache_ttl.pop(key, None)
        else:
            self._cache.clear()
            self._cache_ttl.clear()
        
        logger.info(f"Cleared cache for {query_type or 'all queries'}")
    
    def get_query_history(self, limit: int = 100) -> list[Query]:
        """Get query history."""
        return self._query_history[-limit:]















