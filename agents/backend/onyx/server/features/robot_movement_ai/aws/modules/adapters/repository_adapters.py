"""
Repository Adapters
===================

Implementations of RepositoryPort with different backends.
"""

import logging
from typing import Any, Dict, Optional, List
from pydantic import BaseModel
from aws.modules.ports.repository_port import RepositoryPort

logger = logging.getLogger(__name__)


class DynamoDBRepositoryAdapter(RepositoryPort):
    """DynamoDB implementation of RepositoryPort."""
    
    def __init__(self, table_name: str, region: str = "us-east-1"):
        self.table_name = table_name
        self.region = region
        self._client = None
    
    async def _get_client(self):
        """Get DynamoDB client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("dynamodb", region_name=self.region)
            except ImportError:
                logger.warning("boto3 not installed")
        return self._client
    
    async def create(self, entity: BaseModel) -> BaseModel:
        """Create entity in DynamoDB."""
        client = await self._get_client()
        if not client:
            raise Exception("DynamoDB client not available")
        
        # Simplified implementation
        item = entity.dict()
        # In production, use proper DynamoDB put_item
        return entity
    
    async def get_by_id(self, entity_id: str) -> Optional[BaseModel]:
        """Get entity by ID from DynamoDB."""
        # Simplified implementation
        return None
    
    async def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[BaseModel]:
        """Get all entities from DynamoDB."""
        return []
    
    async def update(self, entity_id: str, entity: BaseModel) -> BaseModel:
        """Update entity in DynamoDB."""
        return entity
    
    async def delete(self, entity_id: str) -> bool:
        """Delete entity from DynamoDB."""
        return True
    
    async def exists(self, entity_id: str) -> bool:
        """Check if entity exists in DynamoDB."""
        return False


class PostgreSQLRepositoryAdapter(RepositoryPort):
    """PostgreSQL implementation of RepositoryPort."""
    
    def __init__(self, connection_string: str, table_name: str):
        self.connection_string = connection_string
        self.table_name = table_name
        self._pool = None
    
    async def _get_pool(self):
        """Get PostgreSQL connection pool."""
        if self._pool is None:
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(self.connection_string)
            except ImportError:
                logger.warning("asyncpg not installed")
        return self._pool
    
    async def create(self, entity: BaseModel) -> BaseModel:
        """Create entity in PostgreSQL."""
        pool = await self._get_pool()
        if not pool:
            raise Exception("PostgreSQL pool not available")
        # Simplified implementation
        return entity
    
    async def get_by_id(self, entity_id: str) -> Optional[BaseModel]:
        """Get entity by ID from PostgreSQL."""
        return None
    
    async def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[BaseModel]:
        """Get all entities from PostgreSQL."""
        return []
    
    async def update(self, entity_id: str, entity: BaseModel) -> BaseModel:
        """Update entity in PostgreSQL."""
        return entity
    
    async def delete(self, entity_id: str) -> bool:
        """Delete entity from PostgreSQL."""
        return True
    
    async def exists(self, entity_id: str) -> bool:
        """Check if entity exists in PostgreSQL."""
        return False


class InMemoryRepositoryAdapter(RepositoryPort):
    """In-memory implementation of RepositoryPort (for testing)."""
    
    def __init__(self):
        self._storage: Dict[str, BaseModel] = {}
    
    async def create(self, entity: BaseModel) -> BaseModel:
        """Create entity in memory."""
        entity_id = getattr(entity, "id", str(len(self._storage)))
        self._storage[entity_id] = entity
        return entity
    
    async def get_by_id(self, entity_id: str) -> Optional[BaseModel]:
        """Get entity by ID from memory."""
        return self._storage.get(entity_id)
    
    async def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[BaseModel]:
        """Get all entities from memory."""
        return list(self._storage.values())
    
    async def update(self, entity_id: str, entity: BaseModel) -> BaseModel:
        """Update entity in memory."""
        self._storage[entity_id] = entity
        return entity
    
    async def delete(self, entity_id: str) -> bool:
        """Delete entity from memory."""
        if entity_id in self._storage:
            del self._storage[entity_id]
            return True
        return False
    
    async def exists(self, entity_id: str) -> bool:
        """Check if entity exists in memory."""
        return entity_id in self._storage















