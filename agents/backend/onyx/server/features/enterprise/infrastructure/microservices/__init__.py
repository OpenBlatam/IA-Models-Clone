from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .service_discovery import (
from .service_mesh import (
from .load_balancer import (
from .message_queue import (
from .api_gateway import (
from .config_management import (
from .service_registry import (
from .distributed_tracing import (
from .resilience import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Microservices Infrastructure
============================

Advanced microservices infrastructure with production-ready patterns:
- Service Discovery (Consul, Eureka)
- Service Mesh Integration (Istio, Linkerd)
- Load Balancing (HAProxy, NGINX, Traefik)
- Message Queues (RabbitMQ, Apache Kafka, Redis Streams)
- API Gateway (Kong, Ambassador, Zuul)
- Configuration Management (Consul KV, etcd)
- Service Registry & Health Checks
- Distributed Tracing (Jaeger, Zipkin)
- Circuit Breakers & Bulkheads
- Retry & Timeout Policies
"""

    ServiceDiscoveryManager,
    ConsulServiceDiscovery,
    EurekaServiceDiscovery,
    KubernetesServiceDiscovery
)

    ServiceMeshManager,
    IstioServiceMesh,
    LinkerdServiceMesh,
    ConsulConnectMesh
)

    LoadBalancerManager,
    HAProxyLoadBalancer,
    NginxLoadBalancer,
    TraefikLoadBalancer
)

    MessageQueueManager,
    RabbitMQService,
    KafkaService,
    RedisStreamsService
)

    APIGatewayManager,
    KongGateway,
    AmbassadorGateway,
    ZuulGateway
)

    ConfigurationManager,
    ConsulConfigProvider,
    EtcdConfigProvider,
    KubernetesConfigProvider
)

    ServiceRegistry,
    ServiceInstance,
    ServiceMetadata
)

    TracingManager,
    JaegerTracing,
    ZipkinTracing,
    OpenTelemetryTracing
)

    ResilienceManager,
    BulkheadPattern,
    RetryPolicy,
    TimeoutPolicy
)

__all__ = [
    # Service Discovery
    "ServiceDiscoveryManager",
    "ConsulServiceDiscovery", 
    "EurekaServiceDiscovery",
    "KubernetesServiceDiscovery",
    
    # Service Mesh
    "ServiceMeshManager",
    "IstioServiceMesh",
    "LinkerdServiceMesh", 
    "ConsulConnectMesh",
    
    # Load Balancing
    "LoadBalancerManager",
    "HAProxyLoadBalancer",
    "NginxLoadBalancer",
    "TraefikLoadBalancer",
    
    # Message Queues
    "MessageQueueManager",
    "RabbitMQService",
    "KafkaService",
    "RedisStreamsService",
    
    # API Gateway
    "APIGatewayManager",
    "KongGateway",
    "AmbassadorGateway",
    "ZuulGateway",
    
    # Configuration
    "ConfigurationManager",
    "ConsulConfigProvider",
    "EtcdConfigProvider",
    "KubernetesConfigProvider",
    
    # Service Registry
    "ServiceRegistry",
    "ServiceInstance",
    "ServiceMetadata",
    
    # Distributed Tracing
    "TracingManager",
    "JaegerTracing",
    "ZipkinTracing",
    "OpenTelemetryTracing",
    
    # Resilience Patterns
    "ResilienceManager",
    "BulkheadPattern",
    "RetryPolicy",
    "TimeoutPolicy",
] 