"""
Mock Adapters
=============

Mock implementations for testing.
"""

from typing import Any, Dict, Optional, List
from pydantic import BaseModel
from aws.modules.ports.repository_port import RepositoryPort
from aws.modules.ports.cache_port import CachePort
from aws.modules.ports.messaging_port import MessagingPort


class MockRepository(RepositoryPort):
    """Mock repository for testing."""
    
    def __init__(self):
        self._storage: Dict[str, BaseModel] = {}
        self._operations: List[Dict[str, Any]] = []
    
    async def create(self, entity: BaseModel) -> BaseModel:
        entity_id = getattr(entity, "id", str(len(self._storage)))
        self._storage[entity_id] = entity
        self._operations.append({"operation": "create", "id": entity_id})
        return entity
    
    async def get_by_id(self, entity_id: str) -> Optional[BaseModel]:
        self._operations.append({"operation": "get_by_id", "id": entity_id})
        return self._storage.get(entity_id)
    
    async def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[BaseModel]:
        self._operations.append({"operation": "get_all", "filters": filters})
        return list(self._storage.values())
    
    async def update(self, entity_id: str, entity: BaseModel) -> BaseModel:
        self._storage[entity_id] = entity
        self._operations.append({"operation": "update", "id": entity_id})
        return entity
    
    async def delete(self, entity_id: str) -> bool:
        if entity_id in self._storage:
            del self._storage[entity_id]
            self._operations.append({"operation": "delete", "id": entity_id})
            return True
        return False
    
    async def exists(self, entity_id: str) -> bool:
        return entity_id in self._storage
    
    def clear(self):
        """Clear all data."""
        self._storage.clear()
        self._operations.clear()
    
    def get_operations(self) -> List[Dict[str, Any]]:
        """Get operation history."""
        return self._operations.copy()


class MockCache(CachePort):
    """Mock cache for testing."""
    
    def __init__(self):
        self._storage: Dict[str, Any] = {}
        self._ttl: Dict[str, float] = {}
        import time
        self._time = time
    
    async def get(self, key: str) -> Optional[Any]:
        if key not in self._storage:
            return None
        
        # Check TTL
        if key in self._ttl:
            if self._time.time() > self._ttl[key]:
                del self._storage[key]
                del self._ttl[key]
                return None
        
        return self._storage[key]
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self._storage[key] = value
        if ttl:
            self._ttl[key] = self._time.time() + ttl
        return True
    
    async def delete(self, key: str) -> bool:
        if key in self._storage:
            del self._storage[key]
            if key in self._ttl:
                del self._ttl[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        return key in self._storage
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        if pattern:
            import fnmatch
            keys_to_delete = [k for k in self._storage.keys() if fnmatch.fnmatch(k, pattern)]
            for key in keys_to_delete:
                await self.delete(key)
            return len(keys_to_delete)
        else:
            count = len(self._storage)
            self._storage.clear()
            self._ttl.clear()
            return count


class MockMessaging(MessagingPort):
    """Mock messaging for testing."""
    
    def __init__(self):
        self._messages: Dict[str, List[Dict[str, Any]]] = {}
        self._subscribers: Dict[str, List] = {}
    
    async def publish(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> bool:
        if topic not in self._messages:
            self._messages[topic] = []
        self._messages[topic].append({"key": key, "message": message})
        
        # Notify subscribers
        if topic in self._subscribers:
            for handler in self._subscribers[topic]:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
        
        return True
    
    async def subscribe(self, topic: str, handler) -> bool:
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(handler)
        return True
    
    async def unsubscribe(self, topic: str) -> bool:
        if topic in self._subscribers:
            del self._subscribers[topic]
            return True
        return False
    
    def get_messages(self, topic: str) -> List[Dict[str, Any]]:
        """Get published messages for topic."""
        return self._messages.get(topic, []).copy()
    
    def clear(self):
        """Clear all messages."""
        self._messages.clear()
        self._subscribers.clear()


# Import asyncio
import asyncio















