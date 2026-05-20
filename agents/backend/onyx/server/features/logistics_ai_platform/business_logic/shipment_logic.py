"""Pure functions for shipment business logic"""

from datetime import datetime
import uuid

from models.schemas import (
    ShipmentStatus,
    TrackingEvent,
    Location,
)


def generate_shipment_id() -> str:
    """Generate a unique shipment ID"""
    return f"S{str(uuid.uuid4())[:8].upper()}"


def generate_shipment_reference(shipment_id: str) -> str:
    """Generate shipment reference"""
    date_str = datetime.now().strftime('%Y%m%d')
    return f"SH-{date_str}-{shipment_id[-6:]}"


def generate_tracking_number() -> str:
    """Generate public tracking number"""
    return f"TRK{str(uuid.uuid4())[:12].upper().replace('-', '')}"


def generate_house_bill_number() -> str:
    """Generate House Bill of Lading number"""
    date_str = datetime.now().strftime('%Y%m%d')
    return f"HBL{date_str}{str(uuid.uuid4())[:8].upper().replace('-', '')}"


def generate_master_bill_number() -> str:
    """Generate Master Bill of Lading number"""
    date_str = datetime.now().strftime('%Y%m%d')
    return f"MBL{date_str}{str(uuid.uuid4())[:8].upper().replace('-', '')}"


def create_initial_tracking_event(
    location: Location,
    description: str = "Shipment booked and confirmed"
) -> TrackingEvent:
    """Create initial tracking event"""
    return TrackingEvent(
        event_type="BOOKED",
        location=location,
        timestamp=datetime.now(),
        description=description,
        status=ShipmentStatus.BOOKED
    )

