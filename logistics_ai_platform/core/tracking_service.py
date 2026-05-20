"""Tracking service for real-time shipment tracking"""

from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging

from models.schemas import (
    ShipmentResponse,
    ContainerResponse,
    TrackingEvent,
    ShipmentStatus,
    Location,
    GPSLocation,
)
from core.shipment_service import ShipmentService
from core.container_service import ContainerService

logger = logging.getLogger(__name__)


class TrackingService:
    """Service for tracking shipments and containers"""
    
    def __init__(
        self,
        shipment_service: ShipmentService,
        container_service: ContainerService
    ):
        """Initialize tracking service"""
        self.shipment_service = shipment_service
        self.container_service = container_service
    
    async def get_tracking_info(
        self,
        shipment_id: Optional[str] = None,
        container_id: Optional[str] = None
    ) -> Dict:
        """Get comprehensive tracking information"""
        result = {
            "shipment": None,
            "containers": [],
            "current_location": None,
            "next_milestone": None,
            "estimated_arrival": None,
            "status": None
        }
        
        if shipment_id:
            shipment = await self.shipment_service.get_shipment(shipment_id)
            if shipment:
                result["shipment"] = shipment
                result["status"] = shipment.status
                result["estimated_arrival"] = shipment.estimated_arrival
                
                # Get containers
                containers = await self.container_service.get_containers_by_shipment(shipment_id)
                result["containers"] = containers
                
                # Get current location from latest tracking event
                if shipment.tracking_events:
                    latest_event = shipment.tracking_events[-1]
                    result["current_location"] = latest_event.location
                
                # Determine next milestone
                result["next_milestone"] = self._get_next_milestone(shipment)
        
        elif container_id:
            container = await self.container_service.get_container(container_id)
            if container:
                result["containers"] = [container]
                result["status"] = container.status
                if container.location:
                    result["current_location"] = container.location
                if container.gps_location:
                    result["current_location"] = Location(
                        country="",
                        city="",
                        latitude=container.gps_location.latitude,
                        longitude=container.gps_location.longitude
                    )
        
        return result
    
    async def get_tracking_history(
        self,
        shipment_id: str
    ) -> List[TrackingEvent]:
        """Get tracking history for a shipment"""
        shipment = await self.shipment_service.get_shipment(shipment_id)
        if not shipment:
            return []
        
        return shipment.tracking_events
    
    async def add_tracking_event(
        self,
        shipment_id: str,
        event_type: str,
        location: Location,
        description: str,
        status: ShipmentStatus,
        metadata: Optional[Dict] = None
    ) -> Optional[ShipmentResponse]:
        """Add a tracking event to a shipment"""
        shipment = await self.shipment_service.get_shipment(shipment_id)
        if not shipment:
            return None
        
        event = TrackingEvent(
            event_type=event_type,
            location=location,
            timestamp=datetime.now(),
            description=description,
            status=status,
            metadata=metadata or {}
        )
        
        shipment.tracking_events.append(event)
        shipment.status = status
        shipment.updated_at = datetime.now()
        
        return shipment
    
    def _get_next_milestone(self, shipment: ShipmentResponse) -> Optional[str]:
        """Determine next milestone for a shipment"""
        status = shipment.status
        
        if status == ShipmentStatus.PENDING:
            return "Awaiting booking confirmation"
        elif status == ShipmentStatus.QUOTED:
            return "Awaiting booking"
        elif status == ShipmentStatus.BOOKED:
            return "Preparing for departure"
        elif status == ShipmentStatus.IN_TRANSIT:
            return "In transit to destination"
        elif status == ShipmentStatus.IN_CUSTOMS:
            return "Clearing customs"
        elif status == ShipmentStatus.DELIVERED:
            return "Delivered"
        else:
            return None
    
    async def get_shipments_by_status(
        self,
        status: ShipmentStatus
    ) -> List[ShipmentResponse]:
        """Get all shipments with a specific status"""
        return await self.shipment_service.get_shipments(status=status)
    
    async def get_departing_this_week(self) -> List[ShipmentResponse]:
        """Get shipments departing this week"""
        all_shipments = await self.shipment_service.get_shipments()
        now = datetime.now()
        week_end = now + timedelta(days=7)
        
        return [
            s for s in all_shipments
            if s.estimated_departure
            and now <= s.estimated_departure <= week_end
        ]
    
    async def get_arriving_this_week(self) -> List[ShipmentResponse]:
        """Get shipments arriving this week"""
        all_shipments = await self.shipment_service.get_shipments()
        now = datetime.now()
        week_end = now + timedelta(days=7)
        
        return [
            s for s in all_shipments
            if s.estimated_arrival
            and now <= s.estimated_arrival <= week_end
        ]
    
    async def get_in_transit(self) -> List[ShipmentResponse]:
        """Get shipments currently in transit"""
        return await self.shipment_service.get_shipments(
            status=ShipmentStatus.IN_TRANSIT
        )













