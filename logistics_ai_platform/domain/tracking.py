"""Tracking domain logic - pure functions for public tracking"""

from typing import Optional, Dict, Any
from datetime import datetime

from models.schemas import (
    ShipmentResponse,
    ContainerResponse,
    TrackingEvent,
    ShipmentStatus,
    Location,
)
from repositories.shipment_repository import ShipmentRepository
from repositories.container_repository import ContainerRepository
from utils.logger import logger
from utils.exceptions import NotFoundError


async def find_shipment_by_identifier(
    identifier: str,
    shipment_repo: ShipmentRepository
) -> Optional[ShipmentResponse]:
    """Find shipment by any identifier (tracking number, house bill, master bill, shipment_id)"""
    # Try by tracking number
    shipment = await shipment_repo.find_by_tracking_number(identifier)
    if shipment:
        return shipment
    
    # Try by house bill
    shipment = await shipment_repo.find_by_house_bill(identifier)
    if shipment:
        return shipment
    
    # Try by master bill
    shipment = await shipment_repo.find_by_master_bill(identifier)
    if shipment:
        return shipment
    
    # Try by shipment_id
    shipment = await shipment_repo.find_by_id(identifier)
    if shipment:
        return shipment
    
    return None


async def find_container_by_number(
    container_number: str,
    container_repo: ContainerRepository
) -> Optional[ContainerResponse]:
    """Find container by container number"""
    return await container_repo.find_by_container_number(container_number)


async def get_public_tracking_info(
    identifier: str,
    shipment_repo: ShipmentRepository,
    container_repo: ContainerRepository
) -> Dict[str, Any]:
    """Get public tracking information by any identifier"""
    # Try to find shipment first
    shipment = await find_shipment_by_identifier(identifier, shipment_repo)
    
    if shipment:
        # Get containers for this shipment
        containers = await container_repo.find_by_shipment_id(shipment.shipment_id)
        
        # Get current location from latest tracking event
        current_location = None
        if shipment.tracking_events:
            latest_event = shipment.tracking_events[-1]
            current_location = latest_event.location
        
        return {
            "found": True,
            "type": "shipment",
            "shipment": {
                "shipment_id": shipment.shipment_id,
                "tracking_number": shipment.tracking_number,
                "house_bill_number": shipment.house_bill_number,
                "master_bill_number": shipment.master_bill_number,
                "shipment_reference": shipment.shipment_reference,
                "status": shipment.status.value,
                "origin": shipment.origin,
                "destination": shipment.destination,
                "carrier": shipment.carrier,
                "estimated_departure": shipment.estimated_departure.isoformat() if shipment.estimated_departure else None,
                "estimated_arrival": shipment.estimated_arrival.isoformat() if shipment.estimated_arrival else None,
                "actual_departure": shipment.actual_departure.isoformat() if shipment.actual_departure else None,
                "actual_arrival": shipment.actual_arrival.isoformat() if shipment.actual_arrival else None,
                "current_location": current_location.model_dump() if current_location else None,
                "next_milestone": _get_next_milestone(shipment),
            },
            "containers": [
                {
                    "container_id": c.container_id,
                    "container_number": c.container_number,
                    "container_type": c.container_type,
                    "status": c.status.value,
                    "location": c.location.model_dump() if c.location else None,
                }
                for c in containers
            ],
            "tracking_events": [
                {
                    "event_type": e.event_type,
                    "timestamp": e.timestamp.isoformat(),
                    "location": e.location.model_dump() if e.location else None,
                    "description": e.description,
                    "status": e.status.value if e.status else None,
                }
                for e in shipment.tracking_events
            ],
        }
    
    # Try to find container
    container = await find_container_by_number(identifier, container_repo)
    
    if container:
        # Get shipment if available
        shipment = None
        if container.shipment_id:
            shipment = await shipment_repo.find_by_id(container.shipment_id)
        
        return {
            "found": True,
            "type": "container",
            "container": {
                "container_id": container.container_id,
                "container_number": container.container_number,
                "container_type": container.container_type,
                "status": container.status.value,
                "location": container.location.model_dump() if container.location else None,
                "gps_location": container.gps_location.model_dump() if container.gps_location else None,
            },
            "shipment": {
                "shipment_id": shipment.shipment_id,
                "tracking_number": shipment.tracking_number,
                "status": shipment.status.value,
                "origin": shipment.origin.model_dump() if shipment else None,
                "destination": shipment.destination.model_dump() if shipment else None,
            } if shipment else None,
        }
    
    return {
        "found": False,
        "message": f"No shipment or container found with identifier: {identifier}",
    }


def _get_next_milestone(shipment: ShipmentResponse) -> Optional[str]:
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








