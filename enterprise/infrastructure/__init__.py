from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .cache import MultiTierCacheService
from .monitoring import PrometheusMetricsService
from .security import CircuitBreakerService
from .health import HealthCheckService
from .rate_limit import RedisRateLimitService
from .microservices import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Infrastructure Layer
===================

Concrete implementations of external concerns like caching, monitoring, etc.
This layer depends on the core interfaces but implements the actual functionality.
"""


# 🚀 NEW MICROSERVICES INFRASTRUCTURE
    # Service Discovery
    ServiceDiscoveryManager,
    ConsulServiceDiscovery,
    ServiceInstance,
    
    # Message Queues
    MessageQueueManager,
    RabbitMQService,
    RedisStreamsService,
    Message,
    
    # Load Balancing
    LoadBalancerManager,
    RoundRobinStrategy,
    WeightedRoundRobinStrategy,
    LeastConnectionsStrategy,
    HealthBasedStrategy,
    
    # Resilience Patterns
    ResilienceManager,
    BulkheadPattern,
    RetryPolicy,
    TimeoutPolicy,
    RetryStrategy,
    
    # Configuration Management
    ConfigurationManager,
    ConsulConfigProvider,
    EnvironmentConfigProvider,
    FileConfigProvider,
    ConfigurationItem,
)

__all__ = [
    # Original services
    "MultiTierCacheService",
    "PrometheusMetricsService",
    "CircuitBreakerService", 
    "HealthCheckService",
    "RedisRateLimitService",
    
    # Microservices Infrastructure
    "ServiceDiscoveryManager",
    "ConsulServiceDiscovery",
    "ServiceInstance",
    "MessageQueueManager",
    "RabbitMQService", 
    "RedisStreamsService",
    "Message",
    "LoadBalancerManager",
    "RoundRobinStrategy",
    "WeightedRoundRobinStrategy",
    "LeastConnectionsStrategy",
    "HealthBasedStrategy",
    "ResilienceManager",
    "BulkheadPattern",
    "RetryPolicy",
    "TimeoutPolicy",
    "RetryStrategy",
    "ConfigurationManager",
    "ConsulConfigProvider",
    "EnvironmentConfigProvider",
    "FileConfigProvider",
    "ConfigurationItem",
] 