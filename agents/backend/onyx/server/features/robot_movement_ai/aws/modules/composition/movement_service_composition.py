"""
Movement Service Composition
============================

Composes movement service from modular components.
"""

import logging
from typing import Dict, Any
from fastapi import FastAPI, Depends
from aws.modules.composition.service_composer import ServiceComposer
from aws.modules.presentation.api_router import APIRouter
from aws.modules.presentation.endpoint_builder import EndpointBuilder
from aws.modules.presentation.response_builder import ResponseBuilder
from aws.modules.business.service_factory import ServiceFactory
from aws.modules.business.domain_services import MovementDomainService
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MoveRequest(BaseModel):
    """Move request model."""
    x: float
    y: float
    z: float
    orientation: list = None


def compose_movement_service(config: Dict[str, Any] = None) -> FastAPI:
    """Compose movement service from modules."""
    # Create composer
    if config:
        composer = ServiceComposer.from_config("movement-service", config)
    else:
        composer = ServiceComposer.from_env("movement-service")
    
    # Create business layer
    service_factory = composer.create_business_layer()
    movement_service = service_factory.create_movement_service()
    
    # Create presentation layer
    router = composer.create_presentation_layer("/api/v1")
    builder = EndpointBuilder(router)
    
    # Define endpoints
    def get_movement_service():
        """Dependency to get movement service."""
        return movement_service
    
    async def move_to_handler(request: MoveRequest, service: MovementDomainService = Depends(get_movement_service)):
        """Move robot to position."""
        result = await service.execute(
            "move_to",
            x=request.x,
            y=request.y,
            z=request.z
        )
        return ResponseBuilder.success(data=result)
    
    async def stop_handler(service: MovementDomainService = Depends(get_movement_service)):
        """Stop robot movement."""
        result = await service.execute("stop")
        return ResponseBuilder.success(data=result)
    
    async def status_handler(service: MovementDomainService = Depends(get_movement_service)):
        """Get robot status."""
        result = await service.execute("get_status")
        return ResponseBuilder.success(data=result)
    
    # Register endpoints
    builder.path("/move/to").method("POST").handler(move_to_handler).summary("Move to position").build()
    builder.path("/move/stop").method("POST").handler(stop_handler).summary("Stop movement").build()
    builder.path("/movement/status").method("GET").handler(status_handler).summary("Get status").build()
    
    # Compose service
    return composer.compose_service([router])

