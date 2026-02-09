"""Container routes"""

from fastapi import APIRouter, Depends
from typing import List

from models.schemas import (
    ContainerRequest,
    ContainerResponse,
    ContainerStatus,
    GPSLocation,
)
from utils.dependencies import get_container_repository
from repositories.container_repository import ContainerRepository
from core.container_service import ContainerService
from utils.exceptions import NotFoundError

router = APIRouter(prefix="/containers", tags=["Containers"])


def get_container_service_dep(
    repository: ContainerRepository = Depends(get_container_repository)
) -> ContainerService:
    """Get container service"""
    return ContainerService(repository)


@router.post("", response_model=ContainerResponse, status_code=201)
async def create_container(
    request: ContainerRequest,
    service: ContainerService = Depends(get_container_service_dep)
) -> ContainerResponse:
    """Create a new container"""
    return await service.create_container(request)


@router.get("/{container_id}", response_model=ContainerResponse)
async def get_container(
    container_id: str,
    service: ContainerService = Depends(get_container_service_dep)
) -> ContainerResponse:
    """Get container by ID"""
    container = await service.get_container(container_id)
    if not container:
        raise NotFoundError("Container", container_id)
    return container


@router.get("/shipment/{shipment_id}", response_model=List[ContainerResponse])
async def get_containers_by_shipment(
    shipment_id: str,
    service: ContainerService = Depends(get_container_service_dep)
) -> List[ContainerResponse]:
    """Get containers for a shipment"""
    return await service.get_containers_by_shipment(shipment_id)


@router.patch("/{container_id}/status", response_model=ContainerResponse)
async def update_container_status(
    container_id: str,
    status: ContainerStatus,
    service: ContainerService = Depends(get_container_service_dep)
) -> ContainerResponse:
    """Update container status"""
    container = await service.update_container_status(
        container_id=container_id,
        status=status
    )
    if not container:
        raise NotFoundError("Container", container_id)
    return container


@router.patch("/{container_id}/gps", response_model=ContainerResponse)
async def update_container_gps(
    container_id: str,
    gps_location: GPSLocation,
    service: ContainerService = Depends(get_container_service_dep)
) -> ContainerResponse:
    """Update container GPS location"""
    container = await service.update_container_gps(
        container_id=container_id,
        gps_location=gps_location
    )
    if not container:
        raise NotFoundError("Container", container_id)
    return container
