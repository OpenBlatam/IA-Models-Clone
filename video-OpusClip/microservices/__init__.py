#!/usr/bin/env python3
"""
Microservices Package

Advanced microservices architecture components for the Video-OpusClip API.
"""

from .service_discovery import (
    ServiceStatus,
    ServiceType,
    ServiceInstance,
    ServiceRegistration,
    ServiceQuery,
    ServiceRegistry,
    ServiceDiscoveryClient,
    ServiceLoadBalancer,
    service_registry,
    service_discovery_client,
    service_load_balancer
)

from .circuit_breaker import (
    CircuitState,
    CircuitBreakerError,
    CircuitBreakerConfig,
    CircuitBreakerStats,
    CircuitBreaker,
    CircuitBreakerManager,
    circuit_breaker,
    circuit_breaker_manager,
    video_processor_cb,
    database_cb,
    cache_cb,
    external_api_cb
)

__all__ = [
    # Service Discovery
    'ServiceStatus',
    'ServiceType',
    'ServiceInstance',
    'ServiceRegistration',
    'ServiceQuery',
    'ServiceRegistry',
    'ServiceDiscoveryClient',
    'ServiceLoadBalancer',
    'service_registry',
    'service_discovery_client',
    'service_load_balancer',
    
    # Circuit Breaker
    'CircuitState',
    'CircuitBreakerError',
    'CircuitBreakerConfig',
    'CircuitBreakerStats',
    'CircuitBreaker',
    'CircuitBreakerManager',
    'circuit_breaker',
    'circuit_breaker_manager',
    'video_processor_cb',
    'database_cb',
    'cache_cb',
    'external_api_cb'
]





























