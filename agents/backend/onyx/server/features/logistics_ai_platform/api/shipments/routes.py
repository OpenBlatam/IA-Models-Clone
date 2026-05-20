"""
Shipment routes

This module provides API endpoints for shipment management including:
- Shipment creation
- Shipment retrieval and filtering
- Status updates with tracking events
"""

from typing import Annotated, List, Optional
from fastapi import APIRouter, Depends, status, Query
from pydantic import BaseModel

from models.schemas import (
    ShipmentRequest,
    ShipmentResponse,
    ShipmentStatus,
    Location,
)
from handlers.shipment_handlers import (
    handle_create_shipment,
    handle_get_shipment,
    handle_get_shipments,
    handle_update_shipment_status,
)
from utils.dependencies import get_shipment_repository
from utils.exceptions import ValidationError
from repositories.shipment_repository import ShipmentRepository

router = APIRouter(
    prefix="/shipments",
    tags=["Shipments"],
    responses={
        404: {"description": "Shipment not found"},
        422: {"description": "Validation error"},
        400: {"description": "Business logic error"}
    }
)


ShipmentRepositoryDep = Annotated[ShipmentRepository, Depends(get_shipment_repository)]


class StatusUpdateRequest(BaseModel):
    """Request model for status update"""
    status: ShipmentStatus
    location: Optional[Location] = None
    description: Optional[str] = None


@router.post(
    "",
    response_model=ShipmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new shipment",
    description="Creates a new shipment with origin, destination, and cargo details"
)
async def create_shipment(
    request: ShipmentRequest,
    repository: ShipmentRepositoryDep
) -> ShipmentResponse:
    """
    Create a new shipment
    
    Args:
        request: Shipment creation request
        repository: Injected shipment repository
        
    Returns:
        ShipmentResponse: Created shipment
        
    Raises:
        ValidationError: If request data is invalid
        BusinessLogicError: If shipment cannot be created
    """
    return await handle_create_shipment(request, repository)


@router.get(
    "",
    response_model=List[ShipmentResponse],
    summary="Get shipments with filtering",
    description="Retrieves shipments with optional status filtering and pagination"
)
async def get_shipments(
    status_filter: Optional[ShipmentStatus] = Query(None, alias="status", description="Filter by shipment status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    repository: ShipmentRepositoryDep
) -> List[ShipmentResponse]:
    """
    Get shipments with optional filtering
    
    Args:
        status_filter: Optional status filter
        limit: Maximum number of results (1-1000)
        offset: Number of results to skip
        repository: Injected shipment repository
        
    Returns:
        List of shipments matching the criteria
        
    Raises:
        ValidationError: If pagination parameters are invalid
    """
    return await handle_get_shipments(
        repository,
        status=status_filter,
        limit=limit,
        offset=offset
    )


@router.get(
    "/{shipment_id}",
    response_model=ShipmentResponse,
    summary="Get shipment by ID",
    description="Retrieves a shipment by its unique identifier"
)
async def get_shipment(
    shipment_id: str,
    repository: ShipmentRepositoryDep
) -> ShipmentResponse:
    """
    Get shipment by ID
    
    Args:
        shipment_id: Unique shipment identifier
        repository: Injected shipment repository
        
    Returns:
        ShipmentResponse: Shipment details with tracking events
        
    Raises:
        NotFoundError: If shipment not found
        ValidationError: If shipment_id is invalid
    """
    if not shipment_id or not shipment_id.strip():
        raise ValidationError("Shipment ID is required", field="shipment_id")
    
    return await handle_get_shipment(shipment_id, repository)


@router.patch(
    "/{shipment_id}/status",
    response_model=ShipmentResponse,
    summary="Update shipment status",
    description="Updates shipment status with optional location and description"
)
async def update_shipment_status(
    shipment_id: str,
    update_request: StatusUpdateRequest,
    repository: ShipmentRepositoryDep
) -> ShipmentResponse:
    """
    Update shipment status
    
    Args:
        shipment_id: Unique shipment identifier
        update_request: Status update request with optional location and description
        repository: Injected shipment repository
        
    Returns:
        ShipmentResponse: Updated shipment with new tracking event
        
    Raises:
        NotFoundError: If shipment not found
        BusinessLogicError: If status transition is invalid
        ValidationError: If shipment_id is invalid
    """
    if not shipment_id or not shipment_id.strip():
        raise ValidationError("Shipment ID is required", field="shipment_id")
    
    return await handle_update_shipment_status(
        shipment_id=shipment_id,
        status=update_request.status,
        repository=repository,
        location=update_request.location,
        description=update_request.description
    )

