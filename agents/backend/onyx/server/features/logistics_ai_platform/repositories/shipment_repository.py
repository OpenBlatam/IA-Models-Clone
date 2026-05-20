"""
Shipment repository for data access

This module provides data access layer for shipment operations,
abstracting storage implementation details.
"""

import logging
from typing import Optional, List
from datetime import datetime

from models.schemas import ShipmentResponse, ShipmentStatus
from repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class ShipmentRepository(BaseRepository[ShipmentResponse]):
    """
    Repository for shipment data access
    
    Provides methods for shipment CRUD operations and queries.
    Currently uses in-memory storage, but can be extended to use
    a database or other persistent storage.
    """
    
    def __init__(self):
        """Initialize repository"""
        super().__init__(entity_name="Shipment")
    
    async def find_all(
        self,
        status: Optional[ShipmentStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ShipmentResponse]:
        """Find all shipments with optional filtering"""
        filter_func = None
        if status:
            filter_func = lambda s: s.status == status
        
        return await super().find_all(
            filter_func=filter_func,
            sort_key=lambda x: x.created_at or datetime.min,
            reverse=True,
            limit=limit,
            offset=offset
        )
    
    async def find_by_tracking_number(
        self,
        tracking_number: str
    ) -> Optional[ShipmentResponse]:
        """Find shipment by tracking number (case-insensitive)"""
        return await self.find_by_field("tracking_number", tracking_number, case_sensitive=False)
    
    async def find_by_house_bill(
        self,
        house_bill_number: str
    ) -> Optional[ShipmentResponse]:
        """Find shipment by house bill number (case-insensitive)"""
        return await self.find_by_field("house_bill_number", house_bill_number, case_sensitive=False)
    
    async def find_by_master_bill(
        self,
        master_bill_number: str
    ) -> Optional[ShipmentResponse]:
        """Find shipment by master bill number (case-insensitive)"""
        return await self.find_by_field("master_bill_number", master_bill_number, case_sensitive=False)
    
    async def find_by_booking_id(self, booking_id: str) -> List[ShipmentResponse]:
        """Find shipments by booking ID"""
        return await self.find_all_by_field("booking_id", booking_id)
    
    async def count_by_status(self, status: ShipmentStatus) -> int:
        """Count shipments by status"""
        return await self.count_by_field("status", status)

