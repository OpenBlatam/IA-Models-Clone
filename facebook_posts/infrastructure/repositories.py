from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from typing import Optional, List, Dict, Any
from datetime import datetime
import json
from abc import ABC, abstractmethod
from ..models.facebook_models import FacebookPostEntity, FacebookPostAnalysis
from ..domain.entities import FacebookPostDomainEntity
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Infrastructure - Repository Implementations
"""




class InMemoryPostRepository:
    """Repositorio en memoria para posts."""
    
    def __init__(self) -> Any:
        self._posts: Dict[str, FacebookPostEntity] = {}
        self._analyses: Dict[str, FacebookPostAnalysis] = {}
    
    async def save(self, post: FacebookPostEntity) -> bool:
        """Guardar post en memoria."""
        try:
            self._posts[post.identifier.content_id] = post
            return True
        except Exception:
            return False
    
    async def find_by_id(self, post_id: str) -> Optional[FacebookPostEntity]:
        """Buscar post por ID."""
        return self._posts.get(post_id)
    
    async def find_by_workspace(self, workspace_id: str) -> List[FacebookPostEntity]:
        """Buscar posts por workspace."""
        return [
            post for post in self._posts.values() 
            if post.onyx_workspace_id == workspace_id
        ]
    
    async def find_by_user(self, user_id: str) -> List[FacebookPostEntity]:
        """Buscar posts por usuario."""
        return [
            post for post in self._posts.values()
            if post.onyx_user_id == user_id
        ]
    
    async def update(self, post: FacebookPostEntity) -> bool:
        """Actualizar post."""
        try:
            if post.identifier.content_id in self._posts:
                self._posts[post.identifier.content_id] = post
                return True
            return False
        except Exception:
            return False
    
    async def delete(self, post_id: str) -> bool:
        """Eliminar post."""
        try:
            if post_id in self._posts:
                del self._posts[post_id]
                return True
            return False
        except Exception:
            return False
    
    async def search(self, query: str, filters: Dict[str, Any]) -> List[FacebookPostEntity]:
        """Buscar posts con filtros."""
        results = []
        for post in self._posts.values():
            # Simple text search
            if query.lower() in post.content.text.lower():
                # Apply filters
                if self._matches_filters(post, filters):
                    results.append(post)
        return results
    
    def _matches_filters(self, post: FacebookPostEntity, filters: Dict[str, Any]) -> bool:
        """Verificar si post coincide con filtros."""
        for key, value in filters.items():
            if key == "status" and post.status.value != value:
                return False
            elif key == "workspace_id" and post.onyx_workspace_id != value:
                return False
            elif key == "user_id" and post.onyx_user_id != value:
                return False
        return True


class InMemoryCacheRepository:
    """Repositorio de cache en memoria."""
    
    def __init__(self) -> Any:
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener del cache."""
        cache_entry = self._cache.get(key)
        if cache_entry:
            # Check TTL
            if datetime.now() < cache_entry['expires_at']:
                return cache_entry['value']
            else:
                del self._cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Guardar en cache."""
        try:
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': datetime.now()
            }
            return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Eliminar del cache."""
        try:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
        except Exception:
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Limpiar por patrón."""
        count = 0
        keys_to_delete = []
        for key in self._cache.keys():
            if pattern in key:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self._cache[key]
            count += 1
        
        return count 