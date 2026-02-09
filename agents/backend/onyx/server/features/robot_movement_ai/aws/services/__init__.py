"""
Microservices Package
=====================

All microservices following stateless principles.
"""

from aws.services.base_service import BaseMicroservice, ServiceConfig
from aws.services.service_registry import ServiceRegistry, ServiceInstance, get_service_registry
from aws.services.service_client import ServiceClient, ServiceClientFactory, CircuitBreaker
from aws.services.movement_service import MovementService
from aws.services.trajectory_service import TrajectoryService
from aws.services.chat_service import ChatService
from aws.services.api_gateway import APIGatewayService

__all__ = [
    "BaseMicroservice",
    "ServiceConfig",
    "ServiceRegistry",
    "ServiceInstance",
    "get_service_registry",
    "ServiceClient",
    "ServiceClientFactory",
    "CircuitBreaker",
    "MovementService",
    "TrajectoryService",
    "ChatService",
    "APIGatewayService",
]















