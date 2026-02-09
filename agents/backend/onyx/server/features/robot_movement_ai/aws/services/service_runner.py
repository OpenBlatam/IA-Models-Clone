"""
Service Runner
=============

Utility to run microservices independently.
"""

import asyncio
import logging
import os
import uvicorn
from typing import Optional
from aws.services.base_service import BaseMicroservice, ServiceConfig
from aws.services.service_registry import get_service_registry, ServiceInstance
from aws.services.movement_service import MovementService
from aws.services.trajectory_service import TrajectoryService
from aws.services.chat_service import ChatService
from aws.services.api_gateway import APIGatewayService

logger = logging.getLogger(__name__)


def get_service_by_name(service_name: str) -> Optional[BaseMicroservice]:
    """Get service instance by name."""
    services = {
        "movement-service": MovementService,
        "trajectory-service": TrajectoryService,
        "chat-service": ChatService,
        "api-gateway": APIGatewayService,
    }
    
    service_class = services.get(service_name)
    if not service_class:
        return None
    
    return service_class()


async def register_service(service: BaseMicroservice):
    """Register service with service registry."""
    registry = get_service_registry()
    
    instance = ServiceInstance(
        service_name=service.config.service_name,
        instance_id=f"{service.config.service_name}-{os.getpid()}",
        host=service.config.host,
        port=service.config.port,
        health_check_url=f"http://{service.config.host}:{service.config.port}{service.config.health_check_path}",
        metadata={"pid": os.getpid()}
    )
    
    registry.register(instance)
    logger.info(f"Registered service: {instance.service_name}")


def run_service(service_name: str, host: str = "0.0.0.0", port: Optional[int] = None):
    """Run a microservice."""
    service = get_service_by_name(service_name)
    if not service:
        raise ValueError(f"Unknown service: {service_name}")
    
    if port:
        service.config.port = port
    
    service.config.host = host
    
    # Setup service
    app = service.setup()
    
    # Register with service registry
    asyncio.run(register_service(service))
    
    # Start heartbeat checker
    registry = get_service_registry()
    registry.start_heartbeat_checker()
    
    # Run service
    logger.info(f"Starting {service_name} on {host}:{service.config.port}")
    uvicorn.run(
        app,
        host=host,
        port=service.config.port,
        log_level="info"
    )


if __name__ == "__main__":
    import sys
    
    service_name = sys.argv[1] if len(sys.argv) > 1 else "api-gateway"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    run_service(service_name, port=port)















