"""
Container repository for data access

This module provides data access layer for container operations,
abstracting storage implementation details.
"""

import logging
from typing import Optional, List
from datetime import datetime

from models.schemas import ContainerResponse, ContainerStatus
from repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class ContainerRepository(BaseRepository[ContainerResponse]):
    """
    Repository for container data access
    
    Provides methods for container CRUD operations and queries.
    Currently uses in-memory storage, but can be extended to use
    a database or other persistent storage.
    """
    
    def __init__(self):
        """Initialize repository"""
        super().__init__(entity_name="Container")
    
    async def find_by_shipment_id(self, shipment_id: str) -> List[ContainerResponse]:
        """Find containers by shipment ID"""
        return await self.find_all_by_field("shipment_id", shipment_id)
    
    async def find_by_container_number(
        self,
        container_number: str
    ) -> Optional[ContainerResponse]:
        """Find container by container number (case-insensitive)"""
        return await self.find_by_field("container_number", container_number, case_sensitive=False)
    
    async def find_by_status(
        self,
        status: ContainerStatus,
        limit: int = 100,
        offset: int = 0
    ) -> List[ContainerResponse]:
        """Find containers by status"""
        return await self.find_all(
            filter_func=lambda c: c.status == status,
            sort_key=lambda x: x.created_at or datetime.min,
            reverse=True,
            limit=limit,
            offset=offset
        )
    
    async def count_by_status(self, status: ContainerStatus) -> int:
        """Count containers by status"""
        return await self.count_by_field("status", status)

