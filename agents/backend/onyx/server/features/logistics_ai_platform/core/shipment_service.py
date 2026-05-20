"""
Shipment service for managing shipments

This service provides business logic for shipment management including:
- Shipment creation and validation
- Status updates with tracking events
- Location tracking and updates
"""

from typing import List, Optional
from datetime import datetime
import logging

from models.schemas import (
    ShipmentRequest,
    ShipmentResponse,
    ShipmentStatus,
    TrackingEvent,
    Location,
    QuoteResponse,
    QuoteOption,
)
from repositories.shipment_repository import ShipmentRepository
from business_logic.shipment_logic import (
    generate_shipment_id,
    generate_shipment_reference,
    create_initial_tracking_event,
)
from validators.shipment_validators import validate_shipment_request
from utils.exceptions import NotFoundError, ValidationError, BusinessLogicError

logger = logging.getLogger(__name__)


class ShipmentService:
    """Service for managing shipments"""
    
    def __init__(self, repository: ShipmentRepository):
        """Initialize shipment service"""
        self.repository = repository
    
    async def create_shipment(self, request: ShipmentRequest) -> ShipmentResponse:
        """Create a new shipment"""
        validate_shipment_request(request)
        
        shipment_id = generate_shipment_id()
        shipment_reference = generate_shipment_reference(shipment_id)
        
        shipment = ShipmentResponse(
            shipment_id=shipment_id,
            booking_id=request.booking_id,
            shipment_reference=shipment_reference,
            origin=request.origin,
            destination=request.destination,
            cargo=request.cargo,
            transportation_mode=request.transportation_mode,
            status=ShipmentStatus.PENDING,
            carrier=request.carrier,
            tracking_events=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        await self.repository.save(shipment)
        logger.info(f"Shipment created: {shipment_id}")
        
        return shipment
    
    async def create_shipment_from_booking(
        self,
        booking_id: str,
        quote: QuoteResponse,
        option: QuoteOption,
        shipper_info: dict,
        consignee_info: dict
    ) -> ShipmentResponse:
        """Create shipment from booking"""
        request = ShipmentRequest(
            booking_id=booking_id,
            origin=quote.origin,
            destination=quote.destination,
            cargo=quote.cargo,
            transportation_mode=option.transportation_mode,
            carrier=option.carrier
        )
        
        shipment = await self.create_shipment(request)
        shipment.status = ShipmentStatus.BOOKED
        shipment.estimated_departure = option.estimated_departure
        shipment.estimated_arrival = option.estimated_arrival
        
        # Add initial tracking event
        initial_event = create_initial_tracking_event(quote.origin)
        shipment.tracking_events.append(initial_event)
        
        await self.repository.save(shipment)
        return shipment
    
    async def get_shipment(self, shipment_id: str) -> Optional[ShipmentResponse]:
        """
        Get shipment by ID
        
        Args:
            shipment_id: Shipment identifier
            
        Returns:
            ShipmentResponse if found, None otherwise
        """
        if not shipment_id:
            raise ValidationError("Shipment ID is required", field="shipment_id")
        
        shipment = await self.repository.find_by_id(shipment_id)
        
        if not shipment:
            logger.debug(f"Shipment not found: {shipment_id}")
        
        return shipment
    
    async def get_shipments(
        self,
        status: Optional[ShipmentStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ShipmentResponse]:
        """Get shipments with optional filtering"""
        return await self.repository.find_all(status=status, limit=limit, offset=offset)
    
    async def update_shipment_status(
        self,
        shipment_id: str,
        status: ShipmentStatus,
        location: Optional[Location] = None,
        description: Optional[str] = None
    ) -> ShipmentResponse:
        """
        Update shipment status with tracking event
        
        Args:
            shipment_id: Shipment identifier
            status: New shipment status
            location: Optional location update
            description: Optional event description
            
        Returns:
            Updated shipment
            
        Raises:
            NotFoundError: If shipment not found
            BusinessLogicError: If status transition is invalid
        """
        shipment = await self.get_shipment_or_raise(shipment_id)
        
        # Validate status transition
        if not self._is_valid_status_transition(shipment.status, status):
            raise BusinessLogicError(
                f"Invalid status transition from {shipment.status} to {status}"
            )
        
        # Update status
        old_status = shipment.status
        shipment.status = status
        shipment.updated_at = datetime.now()
        
        # Add tracking event
        event = TrackingEvent(
            event_type=status.value.upper(),
            location=location or shipment.origin,
            timestamp=datetime.now(),
            description=description or f"Status updated to {status.value}",
            status=status
        )
        shipment.tracking_events.append(event)
        
        # Update timestamps based on status
        now = datetime.now()
        if status == ShipmentStatus.IN_TRANSIT and not shipment.actual_departure:
            shipment.actual_departure = now
        elif status == ShipmentStatus.DELIVERED and not shipment.actual_arrival:
            shipment.actual_arrival = now
        
        try:
            await self.repository.save(shipment)
            logger.info(
                f"Shipment {shipment_id} status updated: "
                f"{old_status} -> {status}"
            )
        except Exception as e:
            logger.error(f"Error updating shipment status: {e}", exc_info=True)
            raise BusinessLogicError(f"Failed to update shipment status: {str(e)}")
        
        return shipment
    
    async def get_shipment_or_raise(self, shipment_id: str) -> ShipmentResponse:
        """
        Get shipment by ID or raise NotFoundError
        
        Args:
            shipment_id: Shipment identifier
            
        Returns:
            ShipmentResponse: Found shipment
            
        Raises:
            NotFoundError: If shipment not found
        """
        shipment = await self.get_shipment(shipment_id)
        if not shipment:
            raise NotFoundError("Shipment", shipment_id)
        return shipment
    
    def _is_valid_status_transition(
        self,
        current_status: ShipmentStatus,
        new_status: ShipmentStatus
    ) -> bool:
        """
        Validate status transition
        
        Args:
            current_status: Current shipment status
            new_status: Desired new status
            
        Returns:
            True if transition is valid, False otherwise
        """
        # Define valid transitions
        valid_transitions = {
            ShipmentStatus.PENDING: [
                ShipmentStatus.BOOKED,
                ShipmentStatus.CANCELLED,
                ShipmentStatus.PENDING
            ],
            ShipmentStatus.BOOKED: [
                ShipmentStatus.IN_TRANSIT,
                ShipmentStatus.CANCELLED,
                ShipmentStatus.BOOKED
            ],
            ShipmentStatus.IN_TRANSIT: [
                ShipmentStatus.DELIVERED,
                ShipmentStatus.IN_TRANSIT,
                ShipmentStatus.DELAYED
            ],
            ShipmentStatus.DELIVERED: [ShipmentStatus.DELIVERED],
            ShipmentStatus.CANCELLED: [ShipmentStatus.CANCELLED],
            ShipmentStatus.DELAYED: [
                ShipmentStatus.IN_TRANSIT,
                ShipmentStatus.DELIVERED,
                ShipmentStatus.DELAYED
            ]
        }
        
        allowed = valid_transitions.get(current_status, [])
        return new_status in allowed

