"""
Container service for managing containers

This service provides business logic for container management including:
- Container creation and validation
- Status updates with automatic timestamp management
- GPS location tracking
- Container queries and filtering
"""

from typing import List, Optional
from datetime import datetime
import uuid
import logging

from models.schemas import (
    ContainerRequest,
    ContainerResponse,
    ContainerStatus,
    Location,
    GPSLocation,
)
from repositories.container_repository import ContainerRepository
from utils.exceptions import NotFoundError, ValidationError, BusinessLogicError

logger = logging.getLogger(__name__)


class ContainerService:
    """
    Service for managing containers
    
    Provides business logic for container operations including
    creation, status updates, location tracking, and queries.
    """
    
    def __init__(self, repository: ContainerRepository):
        """
        Initialize container service
        
        Args:
            repository: Container repository for data access
        """
        self.repository = repository
        logger.info("ContainerService initialized")
    
    async def create_container(self, request: ContainerRequest) -> ContainerResponse:
        """
        Create a new container
        
        Args:
            request: Container creation request
            
        Returns:
            ContainerResponse: Created container
            
        Raises:
            ValidationError: If request data is invalid
            BusinessLogicError: If container cannot be created
        """
        # Validate request
        if not request.container_type:
            raise ValidationError("Container type is required", field="container_type")
        
        if not request.shipment_id:
            raise ValidationError("Shipment ID is required", field="shipment_id")
        
        # Generate unique container ID
        container_id = f"CNT{str(uuid.uuid4())[:8].upper()}"
        container_number = request.container_number or f"CONT-{container_id[-8:]}"
        
        # Check if container number already exists
        existing = await self.repository.find_by_container_number(container_number)
        if existing:
            raise BusinessLogicError(
                f"Container with number '{container_number}' already exists"
            )
        
        # Create container
        now = datetime.now()
        container = ContainerResponse(
            container_id=container_id,
            container_number=container_number,
            container_type=request.container_type,
            shipment_id=request.shipment_id,
            status=ContainerStatus.EMPTY,
            created_at=now,
            updated_at=now
        )
        
        try:
            await self.repository.save(container)
            logger.info(
                f"Container created: {container_id} "
                f"(number: {container_number}, type: {request.container_type})"
            )
        except Exception as e:
            logger.error(f"Error creating container: {e}", exc_info=True)
            raise BusinessLogicError(f"Failed to create container: {str(e)}")
        
        return container
    
    async def get_container(self, container_id: str) -> Optional[ContainerResponse]:
        """
        Get container by ID
        
        Args:
            container_id: Container identifier
            
        Returns:
            ContainerResponse if found, None otherwise
        """
        if not container_id:
            raise ValidationError("Container ID is required", field="container_id")
        
        container = await self.repository.find_by_id(container_id)
        
        if not container:
            logger.debug(f"Container not found: {container_id}")
        
        return container
    
    async def get_container_or_raise(self, container_id: str) -> ContainerResponse:
        """
        Get container by ID or raise NotFoundError
        
        Args:
            container_id: Container identifier
            
        Returns:
            ContainerResponse: Found container
            
        Raises:
            NotFoundError: If container not found
        """
        container = await self.get_container(container_id)
        if not container:
            raise NotFoundError("Container", container_id)
        return container
    
    async def get_containers_by_shipment(
        self, 
        shipment_id: str
    ) -> List[ContainerResponse]:
        """
        Get containers for a shipment
        
        Args:
            shipment_id: Shipment identifier
            
        Returns:
            List of containers for the shipment
        """
        if not shipment_id:
            raise ValidationError("Shipment ID is required", field="shipment_id")
        
        containers = await self.repository.find_by_shipment_id(shipment_id)
        logger.debug(f"Found {len(containers)} containers for shipment {shipment_id}")
        
        return containers
    
    async def update_container_status(
        self,
        container_id: str,
        status: ContainerStatus,
        location: Optional[Location] = None,
        gps_location: Optional[GPSLocation] = None
    ) -> ContainerResponse:
        """
        Update container status with optional location updates
        
        Args:
            container_id: Container identifier
            status: New container status
            location: Optional location update
            gps_location: Optional GPS location update
            
        Returns:
            Updated container
            
        Raises:
            NotFoundError: If container not found
            BusinessLogicError: If status transition is invalid
        """
        container = await self.get_container_or_raise(container_id)
        
        # Validate status transition
        if not self._is_valid_status_transition(container.status, status):
            raise BusinessLogicError(
                f"Invalid status transition from {container.status} to {status}"
            )
        
        # Update status
        old_status = container.status
        container.status = status
        container.updated_at = datetime.now()
        
        # Update location if provided
        if location:
            container.location = location
        
        if gps_location:
            container.gps_location = gps_location
        
        # Update timestamps based on status
        now = datetime.now()
        if status == ContainerStatus.LOADED and not container.loaded_at:
            container.loaded_at = now
        elif status == ContainerStatus.IN_TRANSIT and not container.in_transit_at:
            container.in_transit_at = now
        elif status == ContainerStatus.DELIVERED and not container.delivered_at:
            container.delivered_at = now
        
        try:
            await self.repository.save(container)
            logger.info(
                f"Container {container_id} status updated: "
                f"{old_status} -> {status}"
            )
        except Exception as e:
            logger.error(f"Error updating container status: {e}", exc_info=True)
            raise BusinessLogicError(f"Failed to update container status: {str(e)}")
        
        return container
    
    async def update_container_gps(
        self,
        container_id: str,
        gps_location: GPSLocation
    ) -> ContainerResponse:
        """
        Update container GPS location
        
        Args:
            container_id: Container identifier
            gps_location: New GPS location
            
        Returns:
            Updated container
            
        Raises:
            NotFoundError: If container not found
            ValidationError: If GPS location is invalid
        """
        # Validate GPS location
        if not gps_location:
            raise ValidationError("GPS location is required", field="gps_location")
        
        if not (-90 <= gps_location.latitude <= 90):
            raise ValidationError(
                "Latitude must be between -90 and 90",
                field="gps_location.latitude"
            )
        
        if not (-180 <= gps_location.longitude <= 180):
            raise ValidationError(
                "Longitude must be between -180 and 180",
                field="gps_location.longitude"
            )
        
        container = await self.get_container_or_raise(container_id)
        
        container.gps_location = gps_location
        container.updated_at = datetime.now()
        
        try:
            await self.repository.save(container)
            logger.info(
                f"Container {container_id} GPS updated: "
                f"({gps_location.latitude}, {gps_location.longitude})"
            )
        except Exception as e:
            logger.error(f"Error updating container GPS: {e}", exc_info=True)
            raise BusinessLogicError(f"Failed to update container GPS: {str(e)}")
        
        return container
    
    def _is_valid_status_transition(
        self,
        current_status: ContainerStatus,
        new_status: ContainerStatus
    ) -> bool:
        """
        Validate status transition
        
        Args:
            current_status: Current container status
            new_status: Desired new status
            
        Returns:
            True if transition is valid, False otherwise
        """
        # Define valid transitions
        valid_transitions = {
            ContainerStatus.EMPTY: [
                ContainerStatus.LOADED,
                ContainerStatus.EMPTY
            ],
            ContainerStatus.LOADED: [
                ContainerStatus.IN_TRANSIT,
                ContainerStatus.LOADED
            ],
            ContainerStatus.IN_TRANSIT: [
                ContainerStatus.DELIVERED,
                ContainerStatus.IN_TRANSIT,
                ContainerStatus.DAMAGED
            ],
            ContainerStatus.DELIVERED: [ContainerStatus.DELIVERED],
            ContainerStatus.DAMAGED: [ContainerStatus.DAMAGED]
        }
        
        allowed = valid_transitions.get(current_status, [])
        return new_status in allowed

