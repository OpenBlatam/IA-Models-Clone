"""
Microservices Module
Service discovery, inter-service communication, and API Gateway
"""

from .service_discovery import (
    ServiceRegistry,
    ServiceInstance,
    ServiceStatus,
    get_service_registry
)
from .service_client import (
    ServiceClient,
    ServiceClientPool,
    get_service_client
)
from .api_gateway import (
    APIGateway,
    APIGatewayMiddleware,
    RateLimitConfig,
    get_api_gateway
)
from .event_bus import (
    EventBus,
    Event,
    EventType,
    get_event_bus
)
from .load_balancer import (
    LoadBalancer,
    LoadBalancingStrategy,
    get_load_balancer
)
from .service_mesh import (
    ServiceMesh,
    RequestQueue,
    get_service_mesh
)
from .health_distributed import (
    DistributedHealthChecker,
    get_distributed_health_checker
)
from .graceful_degradation import (
    GracefulDegradation,
    FallbackStrategy,
    DegradationLevel,
    get_graceful_degradation
)

__all__ = [
    "ServiceRegistry",
    "ServiceInstance",
    "ServiceStatus",
    "get_service_registry",
    "ServiceClient",
    "ServiceClientPool",
    "get_service_client",
    "APIGateway",
    "APIGatewayMiddleware",
    "RateLimitConfig",
    "get_api_gateway",
    "EventBus",
    "Event",
    "EventType",
    "get_event_bus",
    "LoadBalancer",
    "LoadBalancingStrategy",
    "get_load_balancer",
    "ServiceMesh",
    "RequestQueue",
    "get_service_mesh",
    "DistributedHealthChecker",
    "get_distributed_health_checker",
    "GracefulDegradation",
    "FallbackStrategy",
    "DegradationLevel",
    "get_graceful_degradation"
]

