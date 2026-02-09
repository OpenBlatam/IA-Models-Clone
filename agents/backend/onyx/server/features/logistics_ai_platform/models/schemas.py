"""Pydantic schemas for Logistics AI Platform"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TransportationMode(str, Enum):
    """Transportation modes"""
    AIR = "air"
    MARITIME = "maritime"
    GROUND = "ground"
    MULTIMODAL = "multimodal"


class ShipmentStatus(str, Enum):
    """Shipment status"""
    PENDING = "pending"
    QUOTED = "quoted"
    BOOKED = "booked"
    IN_TRANSIT = "in_transit"
    IN_CUSTOMS = "in_customs"
    DELIVERED = "delivered"
    DELAYED = "delayed"
    CANCELLED = "cancelled"
    EXCEPTION = "exception"


class ContainerStatus(str, Enum):
    """Container status"""
    EMPTY = "empty"
    LOADED = "loaded"
    IN_TRANSIT = "in_transit"
    AT_PORT = "at_port"
    AT_TERMINAL = "at_terminal"
    DELIVERED = "delivered"
    DAMAGED = "damaged"


class Location(BaseModel):
    """Geographic location"""
    country: str
    city: str
    port_code: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = None


class GPSLocation(BaseModel):
    """GPS location data"""
    latitude: float
    longitude: float
    timestamp: datetime
    accuracy: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None


class CargoDetails(BaseModel):
    """Cargo details"""
    description: str
    weight_kg: float
    volume_m3: Optional[float] = None
    quantity: int
    unit_type: str = Field(default="UNT", description="UNT, PLT, CTN, etc.")
    value_usd: Optional[float] = None
    hs_code: Optional[str] = None
    dangerous_goods: bool = False
    temperature_controlled: bool = False
    special_handling: Optional[str] = None


# Quote Models
class QuoteRequest(BaseModel):
    """Request for a freight quote"""
    origin: Location
    destination: Location
    cargo: CargoDetails
    transportation_mode: TransportationMode
    preferred_departure_date: Optional[datetime] = None
    preferred_arrival_date: Optional[datetime] = None
    insurance_required: bool = False
    special_requirements: Optional[str] = None


class QuoteOption(BaseModel):
    """Quote option"""
    quote_id: str
    transportation_mode: TransportationMode
    carrier: str
    estimated_departure: datetime
    estimated_arrival: datetime
    transit_days: int
    price_usd: float
    currency: str = "USD"
    service_level: str
    features: List[str] = []


class QuoteResponse(BaseModel):
    """Quote response"""
    quote_id: str
    request_id: str
    origin: Location
    destination: Location
    cargo: CargoDetails
    options: List[QuoteOption]
    valid_until: datetime
    created_at: datetime


# Booking Models
class BookingRequest(BaseModel):
    """Booking request"""
    quote_id: str
    selected_option_id: str
    shipper_info: Dict[str, Any]
    consignee_info: Dict[str, Any]
    payment_terms: str
    special_instructions: Optional[str] = None


class BookingResponse(BaseModel):
    """Booking response"""
    booking_id: str
    quote_id: str
    shipment_id: str
    status: ShipmentStatus
    booking_reference: str
    carrier_reference: Optional[str] = None
    created_at: datetime
    estimated_departure: datetime
    estimated_arrival: datetime


# Shipment Models
class ShipmentRequest(BaseModel):
    """Shipment creation request"""
    booking_id: Optional[str] = None
    origin: Location
    destination: Location
    cargo: CargoDetails
    transportation_mode: TransportationMode
    carrier: Optional[str] = None


class TrackingEvent(BaseModel):
    """Tracking event"""
    event_type: str
    location: Location
    timestamp: datetime
    description: str
    status: ShipmentStatus
    metadata: Dict[str, Any] = {}


class ShipmentResponse(BaseModel):
    """Shipment response"""
    shipment_id: str
    booking_id: Optional[str] = None
    shipment_reference: str
    tracking_number: Optional[str] = Field(None, description="Public tracking number")
    house_bill_number: Optional[str] = Field(None, description="House Bill of Lading number")
    master_bill_number: Optional[str] = Field(None, description="Master Bill of Lading number")
    origin: Location
    destination: Location
    cargo: CargoDetails
    transportation_mode: TransportationMode
    status: ShipmentStatus
    carrier: Optional[str] = None
    tracking_events: List[TrackingEvent] = []
    estimated_departure: Optional[datetime] = None
    estimated_arrival: Optional[datetime] = None
    actual_departure: Optional[datetime] = None
    actual_arrival: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


# Container Models
class ContainerRequest(BaseModel):
    """Container request"""
    container_type: str = Field(..., description="20FT, 40FT, 40HC, etc.")
    shipment_id: Optional[str] = None
    container_number: Optional[str] = None


class ContainerResponse(BaseModel):
    """Container response"""
    container_id: str
    container_number: str
    container_type: str
    shipment_id: Optional[str] = None
    status: ContainerStatus
    location: Optional[Location] = None
    gps_location: Optional[GPSLocation] = None
    loaded_at: Optional[datetime] = None
    sealed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


# Tracking Models
class TrackingUpdate(BaseModel):
    """Tracking update"""
    shipment_id: str
    event_type: str
    location: Location
    status: ShipmentStatus
    description: str
    metadata: Dict[str, Any] = {}


# Invoice Models
class InvoiceRequest(BaseModel):
    """Invoice request"""
    shipment_id: str
    invoice_number: Optional[str] = None
    due_date: Optional[datetime] = None
    notes: Optional[str] = None


class InvoiceLineItem(BaseModel):
    """Invoice line item"""
    description: str
    quantity: float
    unit_price: float
    total: float
    tax_rate: Optional[float] = None


class InvoiceResponse(BaseModel):
    """Invoice response"""
    invoice_id: str
    invoice_number: str
    shipment_id: str
    issue_date: datetime
    due_date: Optional[datetime] = None
    subtotal: float
    tax: float
    total: float
    currency: str = "USD"
    line_items: List[InvoiceLineItem]
    status: str
    pdf_url: Optional[str] = None
    created_at: datetime


# Document Models
class DocumentRequest(BaseModel):
    """Document upload request"""
    shipment_id: str
    document_type: str
    file_name: str
    description: Optional[str] = None


class DocumentResponse(BaseModel):
    """Document response"""
    document_id: str
    shipment_id: str
    document_type: str
    file_name: str
    file_url: str
    file_size: int
    mime_type: str
    description: Optional[str] = None
    uploaded_at: datetime


# Alert Models
class AlertRequest(BaseModel):
    """Alert creation request"""
    shipment_id: Optional[str] = None
    container_id: Optional[str] = None
    alert_type: str
    message: str
    priority: str = "medium"  # low, medium, high, critical


class AlertResponse(BaseModel):
    """Alert response"""
    alert_id: str
    shipment_id: Optional[str] = None
    container_id: Optional[str] = None
    alert_type: str
    message: str
    priority: str
    is_read: bool = False
    created_at: datetime
    read_at: Optional[datetime] = None


# Insurance Models
class InsuranceRequest(BaseModel):
    """Insurance request"""
    shipment_id: str
    coverage_type: str = Field(..., description="cargo, container, or both")
    coverage_amount_usd: float
    deductible_usd: Optional[float] = 0.0


class InsuranceResponse(BaseModel):
    """Insurance response"""
    insurance_id: str
    shipment_id: str
    coverage_type: str
    coverage_amount_usd: float
    premium_usd: float
    deductible_usd: float
    policy_number: str
    status: str
    valid_from: datetime
    valid_until: datetime
    created_at: datetime

