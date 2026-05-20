"""
Booking repository for data access

This module provides data access layer for booking operations,
abstracting storage implementation details.
"""

import logging
from typing import Optional, List
from datetime import datetime

from models.schemas import BookingResponse, ShipmentStatus
from repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class BookingRepository(BaseRepository[BookingResponse]):
    """
    Repository for booking data access
    
    Provides methods for booking CRUD operations and queries.
    Currently uses in-memory storage, but can be extended to use
    a database or other persistent storage.
    """
    
    def __init__(self):
        """Initialize repository"""
        super().__init__(entity_name="Booking")
    
    async def find_by_shipment_id(self, shipment_id: str) -> List[BookingResponse]:
        """Find bookings by shipment ID"""
        return await self.find_all_by_field("shipment_id", shipment_id)
    
    async def find_by_quote_id(self, quote_id: str) -> List[BookingResponse]:
        """Find bookings by quote ID"""
        return await self.find_all_by_field("quote_id", quote_id)
    
    async def find_by_booking_reference(
        self,
        booking_reference: str
    ) -> Optional[BookingResponse]:
        """Find booking by booking reference (case-insensitive)"""
        return await self.find_by_field("booking_reference", booking_reference, case_sensitive=False)
    
    async def find_by_status(
        self,
        status: ShipmentStatus,
        limit: int = 100,
        offset: int = 0
    ) -> List[BookingResponse]:
        """Find bookings by status"""
        return await self.find_all(
            filter_func=lambda b: b.status == status,
            sort_key=lambda x: x.created_at or datetime.min,
            reverse=True,
            limit=limit,
            offset=offset
        )
    
    async def count_by_status(self, status: ShipmentStatus) -> int:
        """Count bookings by status"""
        return await self.count_by_field("status", status)

